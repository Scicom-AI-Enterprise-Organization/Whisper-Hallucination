#!/usr/bin/env python3
"""vc_scores.json -> vc_scores_summary.json: drop per-clip detail, keep the aggregates.

`vc_scores.json` carries a `details` list per system (one entry per clip: hypothesis, CER,
similarities) and stays on the box — it is in sync_excludes. The summary is what travels
with the repo and what `plot_vc.py` reads, so it must be regenerated after every re-score.

    .venv_omni/bin/python tts/make_vc_summary.py
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=ROOT / "tts" / "vc_scores.json")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_scores_summary.json")
    args = ap.parse_args()

    scores = json.loads(args.scores.read_text())
    summary = {k: {a: b for a, b in v.items() if a != "details"} for k, v in scores.items()}
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{len(summary)} systems -> {args.out}")
    for k, v in summary.items():
        print(f"  {k:28s} n={v['n']:>4} cer={v['mean_cer']} d_cer={v['mean_d_cer']} "
              f"tgt={v['mean_sim_tgt']} src={v['mean_sim_src']} langs={v['langs']}")


if __name__ == "__main__":
    main()
