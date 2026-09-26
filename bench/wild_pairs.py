#!/usr/bin/env python3
"""The paired question, answered from the wild arm alone.

Does the synthetic lexicon change hallucination and repetition on REAL audio? Two mixes differ
only by those 22,845 clips, so the answer is the gap between the two bars at each configuration.

`eval.json` reports `any_output_rate` on this arm, which counts a bare full stop and sits at
1.000 for nearly everything. That is the wrong reading here. This reports WORDS emitted, which
is the number the rest of the benchmark uses, and reads only `wild.jsonl` so it runs in seconds
rather than re-scoring every arm.

    python bench/wild_pairs.py --json preprint/sweep_rows.json
"""
import argparse, glob, json, os, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer, max_ngram_repeat, normalise  # noqa: E402

LRS = ("1e-4", "2e-4", "5e-4")
RANKS = (8, 16, 32, 64, 128)
# `lora` adapts the attention projections AND fc1/fc2; `loraattn` leaves the MLP alone, which
# is the original LoRA recipe. Same run directory layout, different family prefix.
CONFIGS = ([("lora", r, lr) for r in RANKS for lr in LRS]
           + [("loraattn", r, lr) for r in RANKS for lr in LRS]
           + [("full", 0, lr) for lr in ("5e-6", "1e-5", "2e-5")])
FAM_LABEL = {"lora": "attn+mlp", "loraattn": "attn only", "full": "full"}


def run_name(fam, rank, lr, mix):
    return f"v3_full_{mix}_lr{lr}" if fam == "full" else f"v3_{fam}_r{rank}_{mix}_lr{lr}"


def wild_of(run):
    for cand in [Path(run) / "bench" / "wild.jsonl", *Path(run).glob("bench/*/wild.jsonl")]:
        if cand.exists():
            return [json.loads(l) for l in cand.open(encoding="utf-8")]
    return []


def row_for(run):
    out = {"run": os.path.basename(run)}
    ev = Path(run) / "eval.json"
    if ev.exists():
        d = json.loads(ev.read_text())
        out["lex_rec"] = (d.get("lexicon_synth") or {}).get("recovered_rate")
        out["ls_wer"] = (d.get("librispeech_test_clean") or {}).get("wer")
        out["fl_cer"] = (d.get("fleurs") or {}).get("cer_macro")
        out["fl_med"] = (d.get("fleurs") or {}).get("cer_median_lang")
        for arm in ("silence", "music", "nonspeech"):
            out[arm] = (d.get(arm) or {}).get("hallucination_rate")
        out["rd_runaway"] = (d.get("reduplication") or {}).get("runaway_rate")
        out["rd_empty"] = (d.get("reduplication") or {}).get("empty_rate")
    recs = wild_of(run)
    if recs:
        blank = [r for r in recs if "blank_speech" in ((r.get("meta") or {}).get("reasons") or "")]
        halas = [r for r in recs
                 if "halas_hallucination" in ((r.get("meta") or {}).get("reasons") or "")]
        if blank:
            out["wild_words"] = sum(bool(normalise(r["hyp"])) for r in blank) / len(blank)
            out["wild_loop"] = sum(max_ngram_repeat(r["hyp"], 1) >= 6 for r in blank) / len(blank)
            out["wild_n"] = len(blank)
        if halas:
            out["halas_cer"] = statistics.mean(
                cer(r["reference_text"], r["hyp"]) for r in halas)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--tex", type=Path, default=None,
                    help="emit the paper's paired table as booktabs rows, so no number is "
                         "retyped between the scorer and the manuscript")
    a = ap.parse_args()

    rows = {}
    for d in sorted(glob.glob(f"{a.runs}/*")):
        if os.path.isdir(d):
            rows[os.path.basename(d)] = row_for(d)

    cols = [("wild_words", "wildWd"), ("wild_loop", "wildLp"), ("lex_rec", "lexRec"),
            ("ls_wer", "lsWER"), ("fl_cer", "flCER"), ("silence", "sil"),
            ("rd_empty", "rdEmpt")]
    hdr = f"{'config':<22}" + "".join(f"{lab:>8}" for _, lab in cols)
    print(hdr); print("-" * len(hdr))
    print(f"{'base large-v3':<22}" + "".join(
        f"{v:>8.3f}" for v in (0.999, 0.008, 0.698, 0.035, 0.305, 0.619, 0.001)))
    for fam, rank, lr in CONFIGS:
        for mix in ("no_synth", "all"):
            r = rows.get(run_name(fam, rank, lr, mix))
            if not r:
                continue
            short = {"lora": "L", "loraattn": "A", "full": "F"}[fam]
            tag = (f"{short}r{rank} {lr}" if rank else f"full {lr}") + " " + mix
            print(f"{tag:<22}" + "".join(
                f"{r[k]:>8.3f}" if isinstance(r.get(k), float) else f"{'-':>8}"
                for k, _ in cols))
    if a.tex:
        lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
                 r"\textbf{config} & \textbf{mix} & \textbf{wild words} & \textbf{wild loop}"
                 r" & \textbf{recovered} & \textbf{ls WER} & \textbf{FLEURS CER} \\",
                 r"\midrule",
                 r"\textit{base large-v3} & \textit{none} & 0.999 & 0.008 & 0.698 & 0.035 "
                 r"& 0.305 \\", r"\midrule"]
        for fam, rank, lr in CONFIGS:
            got = False
            for mix in ("no_synth", "all"):
                r = rows.get(run_name(fam, rank, lr, mix))
                if not r or r.get("wild_words") is None:
                    continue
                cfg = ((f"LoRA r{rank} {FAM_LABEL[fam]}, {lr}" if rank
                        else f"full, {lr}") if not got else "")
                got = True
                cells = " & ".join(
                    f"{r[k]:.3f}" if isinstance(r.get(k), float) else "--"
                    for k in ("wild_words", "wild_loop", "lex_rec", "ls_wer", "fl_cer"))
                label = r"\texttt{no\_synth}" if mix == "no_synth" else r"\texttt{all}"
                lines.append(f"{cfg} & {label} & {cells} \\\\")
            if got:
                lines.append(r"\addlinespace[2pt]")
        lines += [r"\bottomrule", r"\end{tabular}"]
        a.tex.parent.mkdir(parents=True, exist_ok=True)
        a.tex.write_text("\n".join(lines) + "\n")
        print(f"-> {a.tex}")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(list(rows.values()), indent=1))
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
