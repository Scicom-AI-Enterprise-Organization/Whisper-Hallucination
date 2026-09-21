"""
reduplication_profile.py
────────────────────────
Slices the `reduplication` arm by the stimulus knobs it was built with — tail silence,
repeat rate, number of repeats, and unit pattern — so the failure rates in
`bench/scores.json` can be read as a function of *what the audio did*, not just a corpus mean.

Per clip, three outcomes from the same definitions the scorer uses (bench/metrics.py):

  runaway   emitted/true repeats > 1.5       (overgeneration_ratio)
  loop      a token run ≥ 6                  (max_ngram_repeat, n=1)
  empty     normalised hypothesis is empty

Aggregated per model over each knob, plus the tail-silence × repeat-count interaction.
Output: bench/reduplication_profile.json, read by bench/plot_reduplication.py.

Runs on the box, where bench/results/ lives:
    python bench/reduplication_profile.py --results bench/results --out bench/reduplication_profile.json
"""
import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import max_ngram_repeat, normalise, overgeneration_ratio  # noqa: E402

KNOBS = ["tail_silence_s", "rate_hz", "n_repeats", "pattern"]


def per_clip(rec, loop_thresh):
    m, hyp = rec["meta"], rec["hyp"]
    ratio = overgeneration_ratio(hyp, m.get("unit", ""), int(m.get("n_repeats", 0)))
    return dict(
        ratio=ratio,
        runaway=ratio > 1.5,
        loop=max_ngram_repeat(hyp, 1) >= loop_thresh,
        empty=not normalise(hyp),
        knobs={k: m[k] for k in KNOBS},
    )


def summarise(clips):
    n = len(clips)
    ratios = sorted(c["ratio"] for c in clips)
    return dict(
        n=n,
        runaway_rate=round(sum(c["runaway"] for c in clips) / n, 4),
        loop_rate=round(sum(c["loop"] for c in clips) / n, 4),
        empty_rate=round(sum(c["empty"] for c in clips) / n, 4),
        overgen_p95=round(ratios[min(n - 1, int(0.95 * n))], 3),
        overgen_mean=round(statistics.mean(ratios), 3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("bench/results"))
    ap.add_argument("--out", type=Path, default=Path("bench/reduplication_profile.json"))
    ap.add_argument("--loop-threshold", type=int, default=6)
    args = ap.parse_args()

    out = {}
    for mdir in sorted(p for p in args.results.glob("*") if p.is_dir()):
        jf = mdir / "reduplication.jsonl"
        if not jf.exists():
            continue
        clips = [per_clip(json.loads(l), args.loop_threshold) for l in jf.open(encoding="utf-8")]
        clips = [c for c in clips if c["knobs"]["pattern"]]
        entry = {"n": len(clips), "overall": summarise(clips), "by": {}, "by_tail_x_repeats": {}}
        for knob in KNOBS:
            groups = defaultdict(list)
            for c in clips:
                groups[c["knobs"][knob]].append(c)
            entry["by"][knob] = {str(k): summarise(v) for k, v in sorted(groups.items())}
        groups = defaultdict(list)
        for c in clips:
            groups[(c["knobs"]["tail_silence_s"], c["knobs"]["n_repeats"])].append(c)
        entry["by_tail_x_repeats"] = {f"{t}|{r}": summarise(v) for (t, r), v in sorted(groups.items())}
        out[mdir.name] = entry
        o = entry["overall"]
        print(f"{mdir.name:40s} n={o['n']} runaway={o['runaway_rate']:.3f} "
              f"loop={o['loop_rate']:.3f} empty={o['empty_rate']:.3f}")

    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
