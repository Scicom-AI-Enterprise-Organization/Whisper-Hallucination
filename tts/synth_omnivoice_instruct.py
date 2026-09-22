#!/usr/bin/env python3
"""OmniVoice's third conditioning path: voice DESIGN, a voice described in words.

`generate()` takes `instruct` alongside `text`/`language`, so a voice can be asked for
("an older man, low pitch, calm") instead of cloned from a reference. That matters here
because the corpus's diversity problem came from OmniVoice's auto mode taking no speaker
argument at all — one voice per language — and the obvious fix, reference cloning, costs
+0.339 CER (`tts/vc_scores_summary.json`). Voice design is the middle path nobody measured:
diverse voices inside the engine that already covers 646 languages, with no reference clip.

Same two outputs as `synth_toucan.py`: a one-clip-per-phrase manifest for the TTS table, and
an all-personas manifest for `voice_diversity.py`.

    .venv_omni/bin/python tts/synth_omnivoice_instruct.py --device cuda:7
"""
import argparse, csv, json, traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
SR_OUT = 16000
FIELDS = ["lang", "phrase", "halluc_count", "audio_filepath", "duration_s", "status",
          "voice", "engine", "error", "meta"]

# `instruct` is a CONTROLLED VOCABULARY, not free text: 48 tags across gender, age, pitch,
# accent and whisper, grouped into mutually exclusive sets (`_INSTRUCT_MUTUALLY_EXCLUSIVE` in
# omnivoice/models/omnivoice.py). Prose like "a young woman with a bright voice" raises
# ValueError rather than being interpreted. One tag per exclusive group, chosen far apart so
# the personas should separate on an x-vector if voice design works at all.
PERSONAS = [
    ("young_female_high", "female, young adult, high pitch"),
    ("elderly_male_low",  "male, elderly, very low pitch"),
    ("middle_female_mod", "female, middle-aged, moderate pitch"),
]


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
    ap.add_argument("--phrases", type=Path, default=Path("tts/out/omnivoice/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("tts/out"))
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--alias", type=Path, default=ROOT / "tts" / "omnivoice_lang_alias.json")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    alias = json.loads(args.alias.read_text()) if args.alias.exists() else {}
    torch.manual_seed(args.seed)
    model = OmniVoice.from_pretrained(args.model)
    model = model.to(args.device).eval() if hasattr(model, "to") else model
    gcfg = OmniVoiceGenerationConfig(num_step=args.num_step)
    sr_model = int(getattr(model, "sample_rate", 0)
                   or getattr(getattr(model, "config", None), "sample_rate", 0) or 24000)

    rows = list(csv.DictReader(args.phrases.open(encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[omnivoice_instruct] {len(rows)} phrases x {len(PERSONAS)} personas @ {sr_model} Hz",
          flush=True)

    single, every = [], []
    for i, r in enumerate(rows):
        lang, phrase = r["lang"], r["phrase"]
        lid = alias.get(lang, lang)
        for pi, (pname, instruct) in enumerate(PERSONAS):
            rec = {"lang": lang, "phrase": phrase, "halluc_count": r.get("halluc_count", ""),
                   "voice": f"omni_design_{pname}", "engine": "omnivoice_instruct",
                   "audio_filepath": "", "duration_s": "", "status": "fail", "error": "",
                   "meta": json.dumps({"tts_model": args.model,
                                       "conditioning": "voice_design_instruct",
                                       "instruct": instruct, "language_id": lid,
                                       "num_step": args.num_step, "seed": args.seed,
                                       "sample_rate_model": sr_model,
                                       "sample_rate_out": SR_OUT})}
            try:
                with torch.no_grad():
                    out = model.generate(text=[phrase], language=[lid], instruct=[instruct],
                                         generation_config=gcfg)
                wav = out[0] if isinstance(out, (list, tuple)) else out
                x = np.asarray(wav, dtype="float32").squeeze()
                if x.size < SR_OUT // 40:
                    raise RuntimeError("output shorter than 25 ms")
                rel = f"wav/{lang}_{i:04d}_p{pi}.flac"
                dur = write_clip(args.out / "omnivoice_instruct_voices" / rel, x, sr_model)
                rec.update(audio_filepath=rel, duration_s=dur, status="ok")
                if pi == 0:
                    s = dict(rec); s["audio_filepath"] = f"../omnivoice_instruct_voices/{rel}"
                    single.append(s)
            except Exception as e:
                traceback.print_exc()
                rec["error"] = f"{type(e).__name__}: {e}"
                if pi == 0:
                    single.append(rec)
            every.append(rec)
        if (i + 1) % 10 == 0:
            ok = sum(x["status"] == "ok" for x in every)
            print(f"  {i+1}/{len(rows)} phrases, {ok}/{len(every)} clips ok", flush=True)

    for name, recs, fname in (("omnivoice_instruct", single, "manifest.csv"),
                              ("omnivoice_instruct_voices", every, "manifest.shard0.csv")):
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        with (d / fname).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader(); w.writerows(recs)
        print(f"[omnivoice_instruct] {sum(r['status'] == 'ok' for r in recs)}/{len(recs)} ok "
              f"-> {d/fname}")


if __name__ == "__main__":
    main()
