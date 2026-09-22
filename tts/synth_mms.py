#!/usr/bin/env python3
"""MMS-TTS over the TTS ablation's phrase set: Meta's 1,100-language VITS models.

One checkpoint PER LANGUAGE (`facebook/mms-tts-<iso3>`), each a single-speaker VITS. That
shape is the point and the limit: coverage nothing else approaches, and exactly one voice per
language -- the same weakness `tts/voice_diversity.py` measured in OmniVoice's auto mode, but
architectural rather than incidental. No reference audio, no speaker argument, no way to ask
for a second voice.

Two of our 22 benchmark languages have no public repo (`urd`, `cmn` return 401), which is
recorded as a failure rather than quietly skipped.

    .venv_mms/bin/python tts/synth_mms.py --device cuda:7
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

# Whisper codes -> the ISO 639-3 code MMS publishes under.
ISO3 = {
    "ar": "ara", "de": "deu", "el": "ell", "en": "eng", "es": "spa", "fr": "fra",
    "he": "heb", "hi": "hin", "id": "ind", "km": "khm", "ko": "kor", "ms": "zlm",
    "pl": "pol", "pt": "por", "ru": "rus", "sw": "swh", "ta": "tam", "th": "tha",
    "tr": "tur", "vi": "vie", "ur": "urd", "zh": "cmn", "bn": "ben",
}


_UROMAN = None


def romanize(text: str) -> str:
    """Latin transliteration for the checkpoints trained on romanised text."""
    global _UROMAN
    if _UROMAN is None:
        import uroman as ur
        _UROMAN = ur.Uroman()
    return _UROMAN.romanize_string(text)


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
    ap.add_argument("--out", type=Path, default=Path("tts/out/mms"))
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer, VitsModel

    rows = list(csv.DictReader(args.phrases.open(encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r)
    print(f"[mms] {len(rows)} phrases over {len(by_lang)} languages", flush=True)

    recs = []
    for lang, items in sorted(by_lang.items()):
        iso = ISO3.get(lang)
        repo = f"facebook/mms-tts-{iso}" if iso else None
        model = tok = None
        err = None
        needs_uroman = False
        if repo is None:
            err = f"no ISO 639-3 mapping for {lang!r}"
        else:
            try:
                tok = AutoTokenizer.from_pretrained(repo)
                model = VitsModel.from_pretrained(repo).to(args.device).eval()
                # Some MMS checkpoints are trained on ROMANISED text and their tokenizer says
                # so via `is_uroman`. Feeding them the native script tokenises to nothing and
                # VITS dies with "narrow(): length must be non-negative" rather than a clear
                # error -- that is what km/ko/hi/zh failed with on the first run.
                needs_uroman = bool(getattr(tok, "is_uroman", False))
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:120]}"
                print(f"  {lang} ({repo}): {err}", flush=True)

        for i, r in enumerate(items):
            rec = {"lang": lang, "phrase": r["phrase"], "halluc_count": r.get("halluc_count", ""),
                   "voice": f"mms_{iso or lang}", "engine": "mms",
                   "audio_filepath": "", "duration_s": "", "status": "fail", "error": err or "",
                   "meta": json.dumps({"tts_model": repo, "conditioning": "per_language_model",
                                       "language_id": iso, "sample_rate_out": SR_OUT})}
            if model is not None:
                try:
                    text = r["phrase"]
                    if needs_uroman:
                        text = romanize(text)
                    inp = tok(text, return_tensors="pt").to(args.device)
                    if inp["input_ids"].shape[-1] == 0:
                        raise RuntimeError("tokenised to zero tokens (script unsupported)")
                    with torch.no_grad():
                        wav = model(**inp).waveform
                    sr = int(getattr(model.config, "sampling_rate", 16000))
                    rel = f"wav/{lang}_{i:04d}.flac"
                    rec.update(audio_filepath=rel, status="ok",
                               duration_s=write_clip(args.out / rel, wav.cpu().numpy(), sr))
                except Exception as e:
                    traceback.print_exc()
                    rec["error"] = f"{type(e).__name__}: {e}"
            recs.append(rec)
        if model is not None:
            del model
            torch.cuda.empty_cache()
        ok = sum(x["status"] == "ok" for x in recs)
        print(f"  {lang}: {ok}/{len(recs)} ok so far", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(recs)
    print(f"[mms] {sum(r['status'] == 'ok' for r in recs)}/{len(recs)} ok -> {args.out}/manifest.csv")


if __name__ == "__main__":
    main()
