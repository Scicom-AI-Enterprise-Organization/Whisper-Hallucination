#!/usr/bin/env python3
"""One harness, several small multilingual TTS models, so each new candidate is a config entry
rather than a new file.

Each model here ships its own Python package and its own idea of an API, so `ENGINES` holds a
loader and a synth callable per model and everything else -- phrase set, manifest schema,
resampling, FLAC writing -- is shared. Models that need a reference clip get one of the
benchmark's own target speakers, so the voice is at least consistent between them.

    .venv_xtts/bin/python      tts/synth_generic.py --engine xtts       --device cuda:7
    .venv_chatterbox/bin/python tts/synth_generic.py --engine chatterbox --device cuda:7
    .venv_voxcpm/bin/python    tts/synth_generic.py --engine voxcpm     --device cuda:7
"""
import argparse, csv, json, traceback
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR_OUT = 16000
FIELDS = ["lang", "phrase", "halluc_count", "audio_filepath", "duration_s", "status",
          "voice", "engine", "error", "meta"]

REF = ROOT / "tts" / "vc_targets" / "en_0194" / "ref_short.flac"   # 6 s LibriSpeech reference


# ── XTTS-v2 ───────────────────────────────────────────────────────────────────────────────
XTTS_LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
              "zh", "hu", "ko", "ja", "hi"}


def load_xtts(device):
    from TTS.api import TTS
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def synth_xtts(model, phrase, lang, device):
    if lang not in XTTS_LANGS:
        raise ValueError(f"xtts does not support {lang!r} (17 languages)")
    wav = model.tts(text=phrase, speaker_wav=str(REF), language=lang)
    return np.asarray(wav, dtype="float32"), 24000


# ── Chatterbox Multilingual ───────────────────────────────────────────────────────────────
CHATTERBOX_LANGS = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
                    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}


def load_chatterbox(device):
    # Chatterbox constructs `perth.PerthImplicitWatermarker()` unconditionally. Installing
    # resemble-perth is not enough: the package sets that name to None when its optional
    # backend is missing, so the call raises "TypeError: 'NoneType' object is not callable"
    # either way. Point it at perth's own DummyWatermarker, which is what we want anyway --
    # a watermark perturbs exactly the audio the scorer measures (tts/vc_openvoice.py kills
    # OpenVoice's for the same reason).
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    return ChatterboxMultilingualTTS.from_pretrained(device=device)


def synth_chatterbox(model, phrase, lang, device):
    if lang not in CHATTERBOX_LANGS:
        raise ValueError(f"chatterbox does not support {lang!r} (23 languages)")
    wav = model.generate(phrase, language_id=lang, audio_prompt_path=str(REF))
    return np.asarray(wav.squeeze().cpu().numpy(), dtype="float32"), int(model.sr)


# ── VoxCPM2 ───────────────────────────────────────────────────────────────────────────────
def load_voxcpm(device):
    from voxcpm import VoxCPM
    return VoxCPM.from_pretrained("openbmb/VoxCPM2")


def synth_voxcpm(model, phrase, lang, device):
    wav = model.generate(text=phrase)
    return np.asarray(np.asarray(wav).squeeze(), dtype="float32"), 16000


# ── Kokoro-82M ────────────────────────────────────────────────────────────────────────────
# lang_code is a single letter, and each one needs its own default voice.
KOKORO = {"en": ("a", "af_heart"), "es": ("e", "ef_dora"), "fr": ("f", "ff_siwis"),
          "hi": ("h", "hf_alpha"), "it": ("i", "if_sara"), "ja": ("j", "jf_alpha"),
          "pt": ("p", "pf_dora"), "zh": ("z", "zf_xiaobei")}


def load_kokoro(device):
    from kokoro import KPipeline
    return {"cls": KPipeline, "cache": {}}


def synth_kokoro(model, phrase, lang, device):
    if lang not in KOKORO:
        raise ValueError(f"kokoro does not support {lang!r} (8 languages)")
    code, voice = KOKORO[lang]
    if code not in model["cache"]:
        model["cache"][code] = model["cls"](lang_code=code)
    chunks = [a for _, _, a in model["cache"][code](phrase, voice=voice)]
    if not chunks:
        raise RuntimeError("kokoro returned no audio")
    return np.concatenate([np.asarray(c, dtype="float32").squeeze() for c in chunks]), 24000


# ── Qwen3-TTS ─────────────────────────────────────────────────────────────────────────────
QWEN_LANGS = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
              "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
              "es": "Spanish", "it": "Italian"}


def load_qwen(device):
    import torch as _t
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                                      device_map=device, dtype=_t.bfloat16)
    speakers = m.get_supported_speakers() if hasattr(m, "get_supported_speakers") else []
    return {"model": m, "speaker": (speakers[0] if speakers else "Vivian")}


def synth_qwen(model, phrase, lang, device):
    if lang not in QWEN_LANGS:
        raise ValueError(f"qwen3-tts does not support {lang!r} (10 languages)")
    wavs, sr = model["model"].generate_custom_voice(
        text=phrase, language=QWEN_LANGS[lang], speaker=model["speaker"])
    return np.asarray(wavs[0], dtype="float32").squeeze(), int(sr)


# ── MOSS-TTS v1.5 ─────────────────────────────────────────────────────────────────────────
def load_moss(device):
    import torch as _t
    from transformers import AutoModel, AutoProcessor
    _t.backends.cuda.enable_cudnn_sdp(False)        # their card: this backend is broken
    mid = "OpenMOSS-Team/MOSS-TTS-v1.5"
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    m = AutoModel.from_pretrained(mid, trust_remote_code=True,
                                  dtype=_t.bfloat16).to(device).eval()
    if hasattr(proc, "audio_tokenizer"):
        proc.audio_tokenizer = proc.audio_tokenizer.to(device)
    return {"model": m, "proc": proc, "device": device}


def synth_moss(model, phrase, lang, device):
    import torch as _t
    proc, m = model["proc"], model["model"]
    conv = [proc.build_user_message(text=phrase, reference=[str(REF)])]
    batch = proc([conv], mode="generation")
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    with _t.no_grad():
        out = m.generate(**batch)
    wav = out[0] if isinstance(out, (list, tuple)) else out
    if hasattr(wav, "cpu"):
        wav = wav.cpu().float().numpy()
    return np.asarray(wav, dtype="float32").squeeze(), 24000


ENGINES = {
    "xtts":       dict(load=load_xtts,       synth=synth_xtts,
                       model_id="coqui/XTTS-v2", conditioning="reference_audio", langs=17),
    "chatterbox": dict(load=load_chatterbox, synth=synth_chatterbox,
                       model_id="ResembleAI/chatterbox", conditioning="reference_audio", langs=23),
    "voxcpm":     dict(load=load_voxcpm,     synth=synth_voxcpm,
                       model_id="openbmb/VoxCPM2", conditioning="default_voice", langs=31),
    "kokoro":     dict(load=load_kokoro,     synth=synth_kokoro,
                       model_id="hexgrad/Kokoro-82M", conditioning="preset_voice", langs=8),
    "qwen3tts":   dict(load=load_qwen,       synth=synth_qwen,
                       model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                       conditioning="preset_voice", langs=10),
    "moss":       dict(load=load_moss,       synth=synth_moss,
                       model_id="OpenMOSS-Team/MOSS-TTS-v1.5",
                       conditioning="reference_audio", langs=31),
}


def write_clip(path: Path, x, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype="float32").squeeze()
    if sr != SR_OUT:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR_OUT)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    sf.write(path, x, SR_OUT, format="FLAC", subtype="PCM_16")
    return round(len(x) / SR_OUT, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--phrases", type=Path, default=Path("tts/out/omnivoice/manifest.csv"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = ENGINES[args.engine]
    out = args.out or Path("tts/out") / args.engine
    rows = list(csv.DictReader(args.phrases.open(encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[{args.engine}] {len(rows)} phrases, {spec['model_id']}", flush=True)

    model = spec["load"](args.device)
    recs = []
    for i, r in enumerate(rows):
        lang, phrase = r["lang"], r["phrase"]
        rec = {"lang": lang, "phrase": phrase, "halluc_count": r.get("halluc_count", ""),
               "voice": args.engine, "engine": args.engine,
               "audio_filepath": "", "duration_s": "", "status": "fail", "error": "",
               "meta": json.dumps({"tts_model": spec["model_id"],
                                   "conditioning": spec["conditioning"],
                                   "reference_audio": REF.name if "reference" in spec["conditioning"] else None,
                                   "sample_rate_out": SR_OUT})}
        try:
            wav, sr = spec["synth"](model, phrase, lang, args.device)
            if wav.size < SR_OUT // 40:
                raise RuntimeError("output shorter than 25 ms")
            rel = f"wav/{lang}_{i:04d}.flac"
            rec.update(audio_filepath=rel, status="ok", duration_s=write_clip(out / rel, wav, sr))
        except Exception as e:
            if not isinstance(e, ValueError):
                traceback.print_exc()
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        recs.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}  {sum(x['status']=='ok' for x in recs)} ok", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(recs)
    ok = sum(r["status"] == "ok" for r in recs)
    print(f"[{args.engine}] {ok}/{len(recs)} ok -> {out}/manifest.csv")


if __name__ == "__main__":
    main()
