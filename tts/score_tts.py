#!/usr/bin/env python3
"""Compare TTS candidates by ASR round-trip: synthesise a phrase, transcribe it, measure CER.

The question is not "which sounds nicer" but "can this model render lexicon phrases
intelligibly in each language" -- because the point of the synthetic audio is to be a
GENUINE positive example of a phrase. If the ASR cannot recover the phrase from the audio,
the clip is useless as a positive no matter how natural it sounds.

Whisper is the judge, with the language FORCED to the target, which is the right call here:
the downstream use is teaching Whisper that this phrase over real speech is legitimate, so
Whisper's own intelligibility is the quantity that matters. It does mean the number is an
ASR-round-trip score, not a human MOS.

    python score_tts.py --systems tts/out/scicom tts/out/higgs --model openai/whisper-large-v3
"""
import argparse, csv, io, json, sys, unicodedata
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
from metrics import cer, normalise  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", type=Path, required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--out", type=Path, default=Path("tts/tts_scores.json"))
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(args.model)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()

    results = {}
    for sysdir in args.systems:
        name = sysdir.name
        rows = list(csv.DictReader((sysdir / "manifest.csv").open(encoding="utf-8")))
        per_lang, details = defaultdict(list), []
        for r in rows:
            if r.get("status") != "ok" or not r["audio_filepath"]:
                per_lang[r["lang"]].append(1.0)          # failure counts as total loss
                details.append({**r, "hyp": "", "cer": 1.0})
                continue
            x, sr = sf.read(sysdir / r["audio_filepath"], dtype="float32")
            feats = proc(x, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                gen = asr.generate(feats.input_features.to(args.device, torch.float16),
                                   language=r["lang"] if len(r["lang"]) == 2 else None,
                                   task="transcribe", max_new_tokens=160, num_beams=1)
            hyp = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
            c = cer(r["phrase"], hyp)
            per_lang[r["lang"]].append(c)
            details.append({**r, "hyp": hyp, "cer": round(c, 4)})
        results[name] = {
            "per_lang_cer": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_lang.items())},
            "mean_cer": round(sum(c for v in per_lang.values() for c in v) /
                              sum(len(v) for v in per_lang.values()), 4),
            "n": len(rows),
            "synth_ok": sum(r.get("status") == "ok" for r in rows),
            "details": details,
        }

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    names = list(results)
    langs = sorted({l for r in results.values() for l in r["per_lang_cer"]})
    w = max(len(n) for n in names) + 2
    print(f"{'lang':<7}" + "".join(f"{n:>{w}}" for n in names) + "   winner")
    print("-" * (7 + w * len(names) + 10))
    wins = defaultdict(int)
    for l in langs:
        vals = [results[n]["per_lang_cer"].get(l) for n in names]
        best = min((v for v in vals if v is not None), default=None)
        win = [n for n, v in zip(names, vals) if v == best]
        wins[win[0]] += 1
        print(f"{l:<7}" + "".join(f"{(f'{v:.3f}' if v is not None else '-'):>{w}}" for v in vals)
              + f"   {win[0]}")
    print("-" * (7 + w * len(names) + 10))
    print(f"{'MEAN':<7}" + "".join(f"{results[n]['mean_cer']:>{w}.3f}" for n in names))
    print(f"{'wins':<7}" + "".join(f"{wins[n]:>{w}}" for n in names))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
