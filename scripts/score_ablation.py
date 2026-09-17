#!/usr/bin/env python3
"""Score ablation results into a comparison table.

Reads ablation/results/<config>/<arm>.jsonl and reports, per (config, arm):

  hall_rate      non-speech arms: share of clips with ANY output. Lower is better;
                 ground truth is the empty string, so any text is a hallucination.
  loop_rate      share of clips whose longest consecutive n-gram run exceeds `--loop-n`.
  max_run        worst consecutive-unigram run seen in the arm.
  overgen_p50/p95  reduplication arm only: predicted repeats / true repeats. 1.0 is
                 correct; >1 is a runaway, <1 means a knob deleted genuine repetition.
  lexicon_rate   share of outputs that are exactly a known hallucination phrase.
  wer            arms that have a reference transcript. This is the regression guard:
                 a config that fixes the non-speech arms but moves WER here is not a win.
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (cer, in_lexicon, load_lexicon, max_ngram_repeat,  # noqa: E402
                     normalise, overgeneration_ratio, wer)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "ablation" / "results"
NONSPEECH_ARMS = {"silence", "musan", "esc50", "nonspeech"}


def score(path: Path, lexicon, loop_n: int, loop_thresh: int, nsp_threshold: float = 0.6) -> dict:
    recs = [json.loads(l) for l in path.open(encoding="utf-8")]
    if not recs:
        return {}
    arm = path.stem
    n = len(recs)
    runs = [max_ngram_repeat(r["hyp"], loop_n) for r in recs]
    row = {
        "arm": arm, "n": n,
        "loop_rate": sum(x >= loop_thresh for x in runs) / n,
        "max_run": max(runs),
        "lexicon_rate": sum(in_lexicon(r["hyp"], lexicon) for r in recs) / n,
    }
    if arm in NONSPEECH_ARMS or all(not r["reference_text"] for r in recs):
        row["hall_rate"] = sum(bool(normalise(r["hyp"])) for r in recs) / n
        # Published HRs disagree by ~10x purely over whether Whisper's own no-speech filter
        # is applied first (Calm-Whisper reports raw; arXiv:2609.04561 reports post-filter).
        # Report both so our numbers are comparable to either.
        kept = [r for r in recs if (r.get("max_no_speech_prob") or 0.0) < nsp_threshold]
        if any(r.get("max_no_speech_prob") is not None for r in recs):
            row["hall_rate_filtered"] = (
                sum(bool(normalise(r["hyp"])) for r in kept) / n)
    refs = [r for r in recs if r["reference_text"]]
    if refs and arm != "reduplication":
        row["wer"] = statistics.mean(wer(r["reference_text"], r["hyp"]) for r in refs)
        row["cer"] = statistics.mean(cer(r["reference_text"], r["hyp"]) for r in refs)
    if arm == "reduplication":
        ratios = [overgeneration_ratio(r["hyp"], r["meta"].get("unit", ""), int(r["meta"].get("n_repeats", 0)))
                  for r in recs if r["meta"].get("unit")]
        if ratios:
            ratios.sort()
            row["overgen_p50"] = statistics.median(ratios)
            row["overgen_p95"] = ratios[min(len(ratios) - 1, int(0.95 * len(ratios)))]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop-n", type=int, default=1, help="n-gram size for the loop check")
    ap.add_argument("--loop-threshold", type=int, default=6, help="consecutive repeats that count as a loop")
    ap.add_argument("--no-speech-threshold", type=float, default=0.6,
                    help="filter used for hall_rate_filtered; Whisper's own default")
    ap.add_argument("--out", type=Path, default=ROOT / "ablation" / "results_summary.csv")
    args = ap.parse_args()

    lex = load_lexicon(ROOT / "lexicon" / "combined_lexicon.csv")
    rows = []
    for cfg_dir in sorted(p for p in RESULTS.glob("*") if p.is_dir()):
        for jf in sorted(cfg_dir.glob("*.jsonl")):
            r = score(jf, lex, args.loop_n, args.loop_threshold, args.no_speech_threshold)
            if r:
                rows.append({"config": cfg_dir.name, **r})
    if not rows:
        raise SystemExit(f"no results under {RESULTS} - run scripts/run_ablation.py first")

    cols = ["config", "arm", "n", "hall_rate", "hall_rate_filtered", "loop_rate", "max_run",
            "overgen_p50", "overgen_p95", "lexicon_rate", "wer", "cer"]
    import csv as _csv
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    width = max(len(r["config"]) for r in rows) + 2
    print(f"{'config':<{width}}{'arm':<15}{'n':>6}{'hall':>8}{'hall_f':>8}{'loop':>8}{'maxrun':>8}{'ovg50':>8}{'ovg95':>8}{'wer':>8}")
    for r in rows:
        fmt = lambda k, p=".3f": f"{r[k]:>8{p}}" if isinstance(r.get(k), float) else f"{'-':>8}"
        print(f"{r['config']:<{width}}{r['arm']:<15}{r['n']:>6}"
              f"{fmt('hall_rate')}{fmt('hall_rate_filtered')}{fmt('loop_rate')}{r.get('max_run','-'):>8}"
              f"{fmt('overgen_p50')}{fmt('overgen_p95')}{fmt('wer')}")
    print(f"\n-> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
