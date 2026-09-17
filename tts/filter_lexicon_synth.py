#!/usr/bin/env python3
"""Score every synthesised clip and keep only the ones that work as positive examples.

Routing by measured CER gets the average right; it does not make any individual clip usable.
A TTS that over-generates on a long phrase emits the phrase plus several seconds of invented
speech, and that clip is a bad positive no matter which engine produced it. So the corpus is
defined by what survives this pass, not by what was generated.

Two gates, both of which a clip must pass:

  cer <= --max-cer                 the ASR recovers the phrase (language FORCED, as in every
                                   other scorer here -- auto-detect on a short non-Latin clip
                                   lands in the wrong language and fails a good clip)
  dur_ratio in [min, max]          the clip is about as long as the phrase should take, so
                                   runaway generation is rejected even when CER survives it

Writes accepted.csv / rejected.csv and a per-language yield report, so a language that
produces nothing usable is visible rather than silently thin.

    .venv_bench/bin/python tts/filter_lexicon_synth.py --device cuda:6
"""
import argparse, csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer, normalise  # noqa: E402

SR = 16000
OUT_FIELDS = ["idx", "lang", "phrase", "count", "engine", "voice", "audio_filepath",
              "duration_s", "cer", "dur_ratio", "hyp", "verdict"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path, default=ROOT / "audio" / "lexicon_synth")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-cer", type=float, default=0.25)
    ap.add_argument("--min-dur-ratio", type=float, default=0.5)
    ap.add_argument("--max-dur-ratio", type=float, default=1.8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "lexicon_synth")
    args = ap.parse_args()

    rows = []
    for man in sorted(args.synth.glob("*/manifest.shard*.csv")):
        with man.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "ok" and r.get("audio_filepath"):
                    r["_dir"] = man.parent
                    rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("nothing to score"); return
    rows.sort(key=lambda r: (r["lang"], r["engine"]))     # forced decode is per batch
    print(f"scoring {len(rows)} clips", flush=True)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(args.model)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()

    acc_fh = (args.out / "accepted.csv").open("w", newline="", encoding="utf-8")
    rej_fh = (args.out / "rejected.csv").open("w", newline="", encoding="utf-8")
    acc = csv.DictWriter(acc_fh, fieldnames=OUT_FIELDS); acc.writeheader()
    rej = csv.DictWriter(rej_fh, fieldnames=OUT_FIELDS); rej.writeheader()

    stats = defaultdict(lambda: {"n": 0, "kept": 0, "cer": [], "reasons": defaultdict(int)})
    batch = []

    def flush(batch):
        if not batch:
            return
        lang = batch[0]["lang"]
        waves = []
        for r in batch:
            x, sr = sf.read(r["_dir"] / r["audio_filepath"], dtype="float32")
            waves.append(x.mean(1) if x.ndim > 1 else x)
        feats = proc(waves, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            gen = asr.generate(feats.input_features.to(args.device, torch.float16),
                               language=lang if len(lang) == 2 else None,
                               task="transcribe", max_new_tokens=220, num_beams=1)
        for r, hyp in zip(batch, proc.batch_decode(gen, skip_special_tokens=True)):
            hyp = hyp.strip()
            c = cer(r["phrase"], hyp)
            expected = max(0.4, len(r["phrase"].split()) / 2.5)
            ratio = float(r["duration_s"]) / expected
            reason = ""
            if c > args.max_cer:
                reason = "cer"
            elif not (args.min_dur_ratio <= ratio <= args.max_dur_ratio):
                reason = "duration"
            elif not normalise(hyp):
                reason = "empty"
            out = {k: r.get(k, "") for k in OUT_FIELDS}
            out.update(cer=round(c, 4), dur_ratio=round(ratio, 3), hyp=hyp[:200],
                       verdict="accept" if not reason else f"reject:{reason}")
            (acc if not reason else rej).writerow(out)
            s = stats[r["lang"]]
            s["n"] += 1; s["cer"].append(c)
            if reason:
                s["reasons"][reason] += 1
            else:
                s["kept"] += 1

    for i, r in enumerate(rows):
        if batch and (r["lang"] != batch[0]["lang"] or len(batch) >= args.batch_size):
            flush(batch); batch = []
        batch.append(r)
        if i and i % (args.batch_size * 40) == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    flush(batch)
    acc_fh.close(); rej_fh.close()

    report = {lang: {"n": s["n"], "kept": s["kept"],
                     "yield": round(s["kept"] / s["n"], 4) if s["n"] else 0.0,
                     "median_cer": round(statistics.median(s["cer"]), 4) if s["cer"] else None,
                     "rejects": dict(s["reasons"])}
              for lang, s in sorted(stats.items())}
    (args.out / "filter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    kept = sum(v["kept"] for v in report.values())
    n = sum(v["n"] for v in report.values())
    print(f"\nkept {kept}/{n} ({kept/max(n,1):.1%}) across {len(report)} languages")
    worst = sorted(report.items(), key=lambda kv: kv[1]["yield"])[:10]
    print("lowest-yield languages:", [(l, f"{v['yield']:.0%}", v["n"]) for l, v in worst])
    print(f"-> {args.out}/accepted.csv, rejected.csv, filter_report.json")


if __name__ == "__main__":
    main()
