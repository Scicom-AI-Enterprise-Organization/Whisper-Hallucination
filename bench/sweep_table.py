#!/usr/bin/env python3
"""One table for the training sweep: both halves of the benchmark, side by side.

A run is only a win if the left block falls and the right block does not. Reporting them
apart invites reading half of it, which is how `blank_only` -- 0.000 runaway, and it deletes
two thirds of repeated speech -- could pass for the best model in the sweep.

Columns marked (wild) are real audio; the rest are built stimuli.

    python bench/sweep_table.py
"""
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer, normalise, max_ngram_repeat, overgeneration_ratio, wer  # noqa: E402


def load(run_bench, arm):
    # run_benchmark nests under the model tag; accept either layout.
    for cand in (Path(run_bench) / f"{arm}.jsonl", *Path(run_bench).glob(f"*/{arm}.jsonl")):
        if cand.exists():
            return [json.loads(l) for l in cand.open(encoding="utf-8")]
    return []


def row_for(run_dir):
    b = Path(run_dir) / "bench"
    out = {"run": os.path.basename(run_dir)}
    for arm in ("silence", "music", "nonspeech"):
        recs = load(b, arm)
        out[arm] = (sum(bool(normalise(r["hyp"])) for r in recs) / len(recs)) if recs else None
    recs = load(b, "reduplication")
    if recs:
        ratios = [overgeneration_ratio(r["hyp"], (r["meta"] or {}).get("unit", ""),
                                       int((r["meta"] or {}).get("n_repeats", 0) or 0))
                  for r in recs]
        out["runaway"] = sum(x > 1.5 for x in ratios) / len(recs)
        out["rd_empty"] = sum(not normalise(r["hyp"]) for r in recs) / len(recs)
    recs = load(b, "lexicon_synth")
    if recs:
        out["lex_rec"] = sum(
            cer(r["reference_text"], r["hyp"]) <= 0.25
            or (len(normalise(r["reference_text"])) <= 12
                and normalise(r["reference_text"]) in normalise(r["hyp"]))
            for r in recs) / len(recs)
    recs = load(b, "wild")
    if recs:
        bs = [r for r in recs if "blank_speech" in ((r["meta"] or {}).get("reasons") or "")]
        hh = [r for r in recs if "halas_hallucination" in ((r["meta"] or {}).get("reasons") or "")]
        if bs:
            out["wild_words"] = sum(bool(normalise(r["hyp"])) for r in bs) / len(bs)
            # The repetition half of the question, on real audio rather than built stimuli.
            out["wild_loop"] = sum(max_ngram_repeat(r["hyp"], 1) >= 6 for r in bs) / len(bs)
        if hh:
            out["halas_cer"] = statistics.mean(cer(r["reference_text"], r["hyp"]) for r in hh)
        lp = [r for r in recs if "loop" in ((r["meta"] or {}).get("reasons") or "")]
        if lp:
            out["wild_loop_mined"] = sum(max_ngram_repeat(r["hyp"], 1) >= 6 for r in lp) / len(lp)
    recs = load(b, "librispeech_test_clean")
    if recs:
        out["ls_wer"] = statistics.mean(wer(r["reference_text"], r["hyp"]) for r in recs)
    out["fl_cer"] = fleurs_macro_cer(load(b, "fleurs"))
    return out


def fleurs_macro_cer(recs):
    """Averaged per LANGUAGE, so 20 clips of Tamil count as much as 20 clips of English.
    CER, not WER: zh/ja/th/lo/my/km do not put spaces between words."""
    if not recs:
        return None
    per = defaultdict(list)
    for r in recs:
        if (r.get("reference_text") or "").strip():
            per[(r.get("meta") or {}).get("lang") or "??"].append(
                cer(r["reference_text"], r["hyp"]))
    if not per:
        return None
    return statistics.mean(statistics.mean(v) for v in per.values())


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None,
                    help="also dump every row here. The run directories live on the GPU box "
                         "and the plots are drawn on the laptop, so the numbers travel as "
                         "JSON rather than installing matplotlib next to the training venv.")
    args = ap.parse_args()

    rows = [row_for(d) for d in sorted(glob.glob("runs/*")) if os.path.isdir(f"{d}/bench")]
    base = json.loads((ROOT / "bench" / "scores.json").read_text()).get("whisper-large-v3", {})
    if base:
        rows.insert(0, {
            "run": "(base large-v3)",
            "silence": base["silence"]["hallucination_rate"],
            "music": base["music"]["hallucination_rate"],
            "nonspeech": base["nonspeech"]["hallucination_rate"],
            "runaway": base["reduplication"]["runaway_rate"],
            "rd_empty": base["reduplication"]["empty_rate"],
            "lex_rec": 0.698, "wild_words": 0.999, "halas_cer": 0.549,
            "ls_wer": base["librispeech_test_clean"]["wer"],
            # The base checkpoint runs the same FLEURS sample through the same harness,
            # into its own directory -- a hardcoded number here would go stale silently.
            "fl_cer": base.get("fleurs", {}).get("cer_macro"),
        })

    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
        print(f"-> {args.json}")

    cols = [("silence", "sil"), ("music", "music"), ("nonspeech", "nonsp"),
            ("runaway", "runawy"), ("rd_empty", "rdEmpt"),
            ("lex_rec", "lexRec"), ("wild_words", "wildWd"), ("halas_cer", "halCER"),
            ("ls_wer", "lsWER"), ("fl_cer", "flCER")]
    hdr = f"{'run':<26}" + "".join(f"{lab:>8}" for _, lab in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['run']:<26}"
        for key, _ in cols:
            v = r.get(key)
            line += f"{v:>8.3f}" if isinstance(v, (int, float)) else f"{'-':>8}"
        print(line)
    print("\nlower is better except lexRec. wildWd/halCER are real audio; the rest are stimuli.")
    print("lsWER is English accuracy, flCER multilingual (FLEURS, per-language mean).")


if __name__ == "__main__":
    main()
