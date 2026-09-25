#!/usr/bin/env python3
"""Score one fine-tuned checkpoint on BOTH halves, in one artifact.

A sweep run is only a win if hallucination and looping fall AND accuracy holds. Reporting
those separately invites reading half the story, so this emits a single `eval.json` per run
with both, plus the deltas against the base checkpoint when one is given.

  hallucination   silence / music / nonspeech -- reference is the empty string, so any output
                  is invented. Reported as words-emitted and as any-output, because published
                  rates differ >10x over exactly that choice (CLAUDE.md).
  repetition      reduplication -- runaway (emitted/true > 1.5), loop (token run >= 6), and
                  empty, where BOTH extremes are wrong.
  lexicon_synth   the phrase IS spoken: recovered rate and CER. The false-positive side.
  wild            real audio, split on `reasons`: blank_speech clips score like the silence
                  arm, HALAS clips score CER against the human-corrected reference, separated
                  by the human verdict.
  accuracy        librispeech WER -- the regression guard.

    python bench/score_eval_run.py --run runs/v3_lora_all --base bench/scores.json
"""
import argparse, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import (cer, in_lexicon, load_lexicon, max_ngram_repeat,  # noqa: E402
                     normalise, overgeneration_ratio, wer)

BLANK_ARMS = ("silence", "music", "nonspeech")


def read(path: Path):
    """`run_benchmark.py` nests its output under the model's tag (`<out>/<tag>/<arm>.jsonl`),
    so a bare `<out>/<arm>.jsonl` is normally absent -- fall back to the nested layout rather
    than silently scoring nothing, which is how the first sweep produced six empty evals."""
    if path.exists():
        return [json.loads(l) for l in path.open(encoding="utf-8")]
    for nested in sorted(path.parent.glob(f"*/{path.name}")):
        return [json.loads(l) for l in nested.open(encoding="utf-8")]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=None, help="default: <run>/bench")
    ap.add_argument("--base", type=Path, default=Path("bench/scores.json"),
                    help="baseline scores.json, for deltas")
    ap.add_argument("--base-model", default="whisper-large-v3")
    ap.add_argument("--lexicon", type=Path, default=ROOT / "lexicon" / "combined_lexicon.csv")
    ap.add_argument("--loop-threshold", type=int, default=6)
    args = ap.parse_args()

    res = args.results or (args.run / "bench")
    lex = load_lexicon(args.lexicon) if args.lexicon.exists() else set()
    out = {"run": str(args.run)}

    # ── hallucination on non-speech ──────────────────────────────────────────────────
    for arm in BLANK_ARMS:
        recs = read(res / f"{arm}.jsonl")
        if not recs:
            continue
        n = len(recs)
        out[arm] = {
            "n": n,
            "hallucination_rate": round(sum(bool(normalise(r["hyp"])) for r in recs) / n, 4),
            "hallucination_rate_any_output": round(sum(bool(r["hyp"].strip()) for r in recs) / n, 4),
            "lexicon_rate": round(sum(in_lexicon(r["hyp"], lex) for r in recs) / n, 4) if lex else None,
        }

    # ── repetition ───────────────────────────────────────────────────────────────────
    recs = read(res / "reduplication.jsonl")
    if recs:
        ratios, runs = [], []
        for r in recs:
            m = r.get("meta") or {}
            ratios.append(overgeneration_ratio(r["hyp"], m.get("unit", ""),
                                               int(m.get("n_repeats", 0) or 0)))
            runs.append(max_ngram_repeat(r["hyp"], 1))
        n = len(recs)
        out["reduplication"] = {
            "n": n,
            "runaway_rate": round(sum(x > 1.5 for x in ratios) / n, 4),
            "loop_rate": round(sum(x >= args.loop_threshold for x in runs) / n, 4),
            "empty_rate": round(sum(not normalise(r["hyp"]) for r in recs) / n, 4),
            "overgen_p95": round(sorted(ratios)[min(n - 1, int(0.95 * n))], 3),
        }

    # ── the false-positive side ──────────────────────────────────────────────────────
    recs = read(res / "lexicon_synth.jsonl")
    if recs:
        cers = [cer(r["reference_text"], r["hyp"]) for r in recs]
        rec_ok = sum(cer(r["reference_text"], r["hyp"]) <= 0.25
                     or (len(normalise(r["reference_text"])) <= 12
                         and normalise(r["reference_text"]) in normalise(r["hyp"]))
                     for r in recs)
        out["lexicon_synth"] = {
            "n": len(recs),
            "recovered_rate": round(rec_ok / len(recs), 4),
            "cer": round(statistics.mean(cers), 4),
            "cer_median": round(statistics.median(cers), 4),
            "empty_rate": round(sum(not normalise(r["hyp"]) for r in recs) / len(recs), 4),
        }

    # ── wild, split on why the clip was collected ────────────────────────────────────
    recs = read(res / "wild.jsonl")
    if recs:
        groups = defaultdict(list)
        for r in recs:
            for reason in ((r.get("meta") or {}).get("reasons") or "unlabelled").split("|"):
                groups[reason].append(r)
        block = {}
        for reason, items in sorted(groups.items()):
            n = len(items)
            b = {"n": n,
                 "any_output_rate": round(sum(bool(r["hyp"].strip()) for r in items) / n, 4),
                 "loop_rate": round(sum(max_ngram_repeat(r["hyp"], 1) >= args.loop_threshold
                                        for r in items) / n, 4)}
            refs = [r for r in items if normalise(r.get("reference_text") or "")]
            if refs:
                b["n_with_reference"] = len(refs)
                b["cer"] = round(statistics.mean(cer(r["reference_text"], r["hyp"]) for r in refs), 4)
            if lex:
                b["lexicon_rate"] = round(sum(in_lexicon(r["hyp"], lex) for r in items) / n, 4)
            block[reason] = b
        out["wild"] = block

    # ── accuracy guard ───────────────────────────────────────────────────────────────
    for arm in ("librispeech_test_clean", "genuine", "speech_in_noise"):
        recs = read(res / f"{arm}.jsonl")
        refs = [r for r in recs if (r.get("reference_text") or "").strip()]
        if refs:
            out[arm] = {"n": len(refs),
                        "wer": round(statistics.mean(wer(r["reference_text"], r["hyp"]) for r in refs), 4),
                        "cer": round(statistics.mean(cer(r["reference_text"], r["hyp"]) for r in refs), 4)}

    # ── deltas against the untuned checkpoint ────────────────────────────────────────
    if args.base.exists():
        base = json.loads(args.base.read_text()).get(args.base_model, {})
        d = {}
        for arm in BLANK_ARMS:
            if arm in out and arm in base:
                d[f"{arm}.hallucination_rate"] = round(
                    out[arm]["hallucination_rate"] - base[arm].get("hallucination_rate", 0), 4)
        if "reduplication" in out and "reduplication" in base:
            for k in ("runaway_rate", "empty_rate"):
                d[f"reduplication.{k}"] = round(
                    out["reduplication"][k] - base["reduplication"].get(k, 0), 4)
        if "librispeech_test_clean" in out and "librispeech_test_clean" in base:
            d["librispeech.wer"] = round(out["librispeech_test_clean"]["wer"]
                                         - base["librispeech_test_clean"].get("wer", 0), 4)
        out["delta_vs_base"] = d

    (args.run / "eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "wild"}, indent=1)[:1400])
    if "wild" in out:
        print("wild:", json.dumps(out["wild"])[:400])
    print(f"-> {args.run}/eval.json")


if __name__ == "__main__":
    main()
