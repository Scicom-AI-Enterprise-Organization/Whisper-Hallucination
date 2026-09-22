#!/usr/bin/env python3
"""ToucanTTS (IMS-Toucan) over the TTS ablation's phrase set — 7,233 languages, and voices
that are SAMPLED rather than cloned.

Why it is worth a row. Every other candidate either covers a handful of languages or, like
OmniVoice, renders one voice per language because it takes no speaker argument (measured:
33 of its 74 languages at a median pairwise cosine of 0.75+, `tts/voice_diversity.py`).
Toucan is the only open model that does both at once: articulatory features give it ~7,000
languages by ISO 639-3, and a WGAN over speaker embeddings (`embedding_gan.pt`) emits
arbitrarily many distinct artificial voices with no reference clip — so diversity costs
nothing in CER, unlike the +0.339 reference cloning charges.

Two outputs, because they answer different questions:

  <out>/toucan/manifest.csv              voice seed 0 only, one clip per phrase — the row that
                                         joins the TTS table, scored by `score_tts.py`
  <out>/toucan_voices/manifest.shard0.csv  every seed — read by `voice_diversity.py`

    .venv_toucan/bin/python tts/synth_toucan.py --device cuda:6 --voices 3
"""
import argparse, csv, json, sys, traceback
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR_OUT = 16000
FIELDS = ["lang", "phrase", "halluc_count", "audio_filepath", "duration_s", "status",
          "voice", "engine", "error", "meta"]

# Whisper-style codes -> ISO 639-3, which is what Toucan indexes languages by. The lexicon
# uses the former everywhere; an unmapped code must FAIL rather than silently fall back, the
# lesson from OmniVoice's language-agnostic mode.
ISO3 = {
    "en": "eng", "zh": "cmn", "de": "deu", "es": "spa", "ru": "rus", "ko": "kor",
    "fr": "fra", "ja": "jpn", "pt": "por", "tr": "tur", "pl": "pol", "ca": "cat",
    "nl": "nld", "ar": "arb", "sv": "swe", "it": "ita", "id": "ind", "hi": "hin",
    "fi": "fin", "vi": "vie", "he": "heb", "uk": "ukr", "el": "ell", "ms": "zsm",
    "cs": "ces", "ro": "ron", "da": "dan", "hu": "hun", "ta": "tam", "no": "nob",
    "th": "tha", "ur": "urd", "hr": "hrv", "bg": "bul", "lt": "lit", "la": "lat",
    "mi": "mri", "ml": "mal", "cy": "cym", "sk": "slk", "te": "tel", "fa": "fas",
    "lv": "lav", "bn": "ben", "sr": "srp", "az": "aze", "sl": "slv", "kn": "kan",
    "et": "est", "mk": "mkd", "br": "bre", "eu": "eus", "is": "isl", "hy": "hye",
    "ne": "npi", "mn": "mon", "bs": "bos", "kk": "kaz", "sq": "sqi", "sw": "swh",
    "gl": "glg", "mr": "mar", "pa": "pan", "si": "sin", "km": "khm", "sn": "sna",
    "yo": "yor", "so": "som", "af": "afr", "oc": "oci", "ka": "kat", "be": "bel",
    "tg": "tgk", "sd": "snd", "gu": "guj", "am": "amh", "yi": "yid", "lo": "lao",
    "uz": "uzb", "fo": "fao", "ht": "hat", "ps": "pbu", "tk": "tuk", "nn": "nno",
    "mt": "mlt", "sa": "san", "lb": "ltz", "my": "mya", "bo": "bod", "tl": "tgl",
    "mg": "plt", "as": "asm", "tt": "tat", "haw": "haw", "ln": "lin", "ha": "hau",
    "ba": "bak", "jw": "jav", "su": "sun", "yue": "yue",
}


def write_clip(path: Path, x: np.ndarray, sr: int) -> float:
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
    ap.add_argument("--phrases", type=Path, default=Path("tts/out/omnivoice/manifest.csv"),
                    help="phrase set to reproduce -- default is the TTS ablation's own "
                         "manifest, so the new row is scored on identical text")
    ap.add_argument("--repo", type=Path, default=Path("vc_repos/IMS-Toucan"))
    ap.add_argument("--out", type=Path, default=Path("tts/out"))
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--voices", type=int, default=3, help="artificial voices sampled per phrase")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.phrases.open(encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[toucan] {len(rows)} phrases x {args.voices} voices", flush=True)

    sys.path.insert(0, str(args.repo.resolve()))
    import os, types
    # ToucanTTSInterface imports `sounddevice` at module level purely so `read_aloud` can play
    # audio through speakers. It needs PortAudio, the box has no sound card and its apt is
    # broken, and we only ever write files -- so satisfy the import with a stub rather than
    # fixing a package manager for a playback library.
    if "sounddevice" not in sys.modules:
        stub = types.ModuleType("sounddevice")
        stub.play = lambda *a, **k: None
        stub.wait = lambda *a, **k: None
        stub.stop = lambda *a, **k: None
        stub.default = types.SimpleNamespace(samplerate=None, channels=None)
        sys.modules["sounddevice"] = stub
    os.chdir(args.repo.resolve())          # it resolves model paths relative to cwd
    from InferenceInterfaces.ControllableInterface import ControllableInterface

    iface = ControllableInterface(gpu_id=args.device.split(":")[-1],
                                  available_artificial_voices=max(args.voices, 10))
    os.chdir(ROOT)

    single, every = [], []
    for i, r in enumerate(rows):
        lang, phrase = r["lang"], r["phrase"]
        iso = ISO3.get(lang)
        for seed in range(args.voices):
            rec = {"lang": lang, "phrase": phrase, "halluc_count": r.get("halluc_count", ""),
                   "voice": f"toucan_gan_{seed}", "engine": "toucan",
                   "audio_filepath": "", "duration_s": "", "status": "fail", "error": "",
                   "meta": json.dumps({"tts_model": "Flux9665/ToucanTTS",
                                       "conditioning": "sampled_speaker_embedding",
                                       "voice_seed": seed, "language_id": iso,
                                       "sample_rate_out": SR_OUT})}
            if iso is None:
                rec["error"] = f"no ISO 639-3 mapping for {lang!r}"
                every.append(rec)
                if seed == 0:
                    single.append(rec)
                continue
            try:
                # ControllableInterface.read returns (sr, wav, matplotlib_figure_path) -- the
                # third element exists for its GUI demo and is discarded here.
                sr, wav, _fig = iface.read(
                    prompt=phrase, reference_audio=None, language=iso, accent=iso,
                    voice_seed=seed, prosody_creativity=0.1,
                    duration_scaling_factor=1.0, pause_duration_scaling_factor=1.0,
                    pitch_variance_scale=1.0, energy_variance_scale=1.0,
                    emb_slider_1=0.0, emb_slider_2=0.0, emb_slider_3=0.0,
                    emb_slider_4=0.0, emb_slider_5=0.0, emb_slider_6=0.0,
                    loudness_in_db=-24.0)
                rel = f"wav/{lang}_{i:04d}_v{seed}.flac"
                dur = write_clip(args.out / "toucan_voices" / rel, wav, sr)
                rec.update(audio_filepath=rel, duration_s=dur, status="ok")
                if seed == 0:
                    # The table row points at the same file, one directory up.
                    s = dict(rec); s["audio_filepath"] = f"../toucan_voices/{rel}"
                    single.append(s)
            except Exception as e:
                traceback.print_exc()
                rec["error"] = f"{type(e).__name__}: {e}"
                if seed == 0:
                    single.append(rec)
            every.append(rec)
        if (i + 1) % 10 == 0:
            ok = sum(x["status"] == "ok" for x in every)
            print(f"  {i+1}/{len(rows)} phrases, {ok}/{len(every)} clips ok", flush=True)

    for name, recs, fname in (("toucan", single, "manifest.csv"),
                              ("toucan_voices", every, "manifest.shard0.csv")):
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        with (d / fname).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader(); w.writerows(recs)
        print(f"[toucan] {sum(r['status'] == 'ok' for r in recs)}/{len(recs)} ok -> {d/fname}")


if __name__ == "__main__":
    main()
