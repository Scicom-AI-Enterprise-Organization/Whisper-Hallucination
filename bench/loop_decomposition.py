#!/usr/bin/env python3
"""Split each arm's WER into the part looping causes and the part that is real error.

A clip that loops emits tokens until `max_new_tokens` and scores a WER in the tens, so a
handful of them dominate a corpus mean. Reporting only the mean therefore says "this model
transcribes badly" when the truth is "this model transcribes well and occasionally runs
away" -- a different problem with a different fix.

    python bench/loop_decomposition.py --results bench/results --out bench/loop_decomposition.json
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import max_ngram_repeat, wer  # noqa: E402

WER_ARMS = ["librispeech_test_clean", "genuine", "genuine_isolated", "speech_in_noise"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("bench/results"))
    ap.add_argument("--out", type=Path, default=Path("bench/loop_decomposition.json"))
    ap.add_argument("--loop-n", type=int, default=1)
    ap.add_argument("--loop-threshold", type=int, default=6)
    args = ap.parse_args()

    out = {}
    for mdir in sorted(p for p in args.results.glob("*") if p.is_dir()):
        per_arm = {}
        for arm in WER_ARMS:
            f = mdir / f"{arm}.jsonl"
            if not f.exists():
                continue
            recs = [json.loads(l) for l in f.open(encoding="utf-8")]
            looped = [max_ngram_repeat(r["hyp"], args.loop_n) >= args.loop_threshold for r in recs]
            clean = [r for r, lp in zip(recs, looped) if not lp]
            per_arm[arm] = {
                "n": len(recs),
                "loop_rate": round(sum(looped) / len(recs), 4),
                "wer": round(statistics.mean(wer(r["reference_text"], r["hyp"]) for r in recs), 4),
                "wer_excluding_looped": round(
                    statistics.mean(wer(r["reference_text"], r["hyp"]) for r in clean), 4) if clean else None,
            }
        if per_arm:
            out[mdir.name] = per_arm

    args.out.write_text(json.dumps(out, indent=2))
    hdr = f"{'model':<36}{'arm':<24}{'looped':>8}{'WER':>8}{'WER|clean':>11}"
    print(hdr); print("-" * len(hdr))
    for m, arms in out.items():
        for a, s in arms.items():
            print(f"{m:<36}{a:<24}{s['loop_rate']:>7.1%}{s['wer']:>8.3f}{s['wer_excluding_looped']:>11.3f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
