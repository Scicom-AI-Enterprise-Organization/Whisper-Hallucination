#!/usr/bin/env python3
"""Render the README's two results tables from the run directories.

Rows filled by hand go stale the moment one run is re-evaluated, and a stale cell in a results
table is worse than an empty one. This prints both markdown blocks; `--write` splices each one
into README.md between its marker comments.

  grid-table   method x learning rate, twelve runs on the `all` mix
  mix-table    the mix sweep: same method, one run per training mix

    python bench/readme_tables.py --write
"""
import argparse, glob, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from sweep_table import row_for, load, fleurs_macro_cer  # noqa: E402
import json  # noqa: E402



# (run dir, method, rank/alpha, params, lr) -- the grid as launched, so a run that failed to
# produce a checkpoint shows as an empty row rather than vanishing from the table.
PARAMS = {32: "57.7 M", 64: "115.3 M", 128: "230.7 M", 0: "1,574.9 M"}
GRID = ([("v3_lora_r{r}_all_lr{lr}", "LoRA", r, lr)
         for r in (32, 64, 128) for lr in ("1e-4", "2e-4", "5e-4")]
        + [("v3_full_all_lr{lr}", "full", 0, lr) for lr in ("5e-6", "1e-5", "2e-5")])

COLS = [("silence", "sil"), ("music", "music"), ("nonspeech", "nonsp"),
        ("runaway", "runaway"), ("rd_empty", "rdEmpty"), ("lex_rec", "lexRec"),
        ("wild_words", "wildWd"), ("halas_cer", "halCER"), ("ls_wer", "lsWER"),
        ("fl_cer", "flCER")]


def fmt(v, best=False):
    if not isinstance(v, (int, float)):
        return "—"
    return f"**{v:.3f}**" if best else f"{v:.3f}"


MIXES = [("v3_lora_blank_only", "`blank_only`"), ("v3_lora_corpus", "`corpus`"),
         ("v3_lora_plus_synth", "`plus_synth`"), ("v3_lora_all", "**`all`**"),
         ("v3_lora_balanced", "`balanced`"), ("v3_lora_synth_heavy", "`synth_heavy`")]


def base_vals():
    base = json.loads((ROOT / "bench" / "scores.json").read_text())["whisper-large-v3"]
    return {
        "silence": base["silence"]["hallucination_rate"],
        "music": base["music"]["hallucination_rate"],
        "nonspeech": base["nonspeech"]["hallucination_rate"],
        "runaway": base["reduplication"]["runaway_rate"],
        "rd_empty": base["reduplication"]["empty_rate"],
        "lex_rec": 0.698, "wild_words": 0.999, "halas_cer": 0.549,
        "ls_wer": base["librispeech_test_clean"]["wer"],
        "fl_cer": base.get("fleurs", {}).get("cer_macro"),
    }


def render(rows, lead_headers):
    """rows: [(lead cells, values dict)]. The first row is the reference and is never bolded --
    bolding the untuned checkpoint would read as 'the base model won'."""
    best = {}
    for key, _ in COLS:
        vs = [v.get(key) for _, v in rows[1:] if isinstance(v.get(key), (int, float))]
        if vs:
            best[key] = max(vs) if key == "lex_rec" else min(vs)
    head = "| " + " | ".join(lead_headers) + " | " + " | ".join(l for _, l in COLS) + " |"
    sep = "|" + "---|" * len(lead_headers) + "---:|" * len(COLS)
    lines = [head, sep]
    for i, (lead, vals) in enumerate(rows):
        cells = [fmt(vals.get(k), i > 0 and best.get(k) is not None and vals.get(k) == best.get(k))
                 for k, _ in COLS]
        lines.append("| " + " | ".join(lead) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def splice(text, name, block):
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    assert begin in text and end in text, f"{name} markers missing from README.md"
    return re.sub(re.escape(begin) + r".*?" + re.escape(end),
                  lambda _: f"{begin}\n\n{block}\n\n{end}", text, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dump", type=Path, default=None,
                    help="write both blocks to a file. The run directories live on the GPU "
                         "box and README.md lives in git on the laptop, so the tables are "
                         "rendered there and spliced here.")
    ap.add_argument("--splice", type=Path, default=None, help="splice a --dump file into README.md")
    a = ap.parse_args()

    if a.splice:
        grid_block, mix_block = a.splice.read_text().split("<!--split-->")
        p = ROOT / "README.md"
        tx = p.read_text()
        tx = splice(tx, "grid-table", grid_block.strip())
        tx = splice(tx, "mix-table", mix_block.strip())
        p.write_text(tx)
        print(f"-> README.md (spliced from {a.splice})")
        return

    def vals_for(d):
        d = Path(a.runs) / d
        return row_for(d) if (d / "bench").is_dir() else {}

    grid = [(["*base large-v3*", "—", "—", "—"], base_vals())]
    for tmpl, method, r, lr in GRID:
        grid.append(([method, f"{r} / {2*r}" if r else "—", PARAMS[r], lr],
                     vals_for(tmpl.format(r=r, lr=lr))))
    grid_block = render(grid, ["method", "rank/α", "params", "lr"])

    mix = [(["*base large-v3*", "—", "—"], base_vals())]
    for run, label in MIXES:
        # The mix sweep ran at 1e-3 first; where a 2e-4 re-run exists it is the honest row,
        # because 1e-3 diverged and a diverged checkpoint is not a data point about the mix.
        for lr, suffix in (("2e-4", "_lr2e4"), ("1e-3", "")):
            d = Path(a.runs) / (run + suffix)
            v = vals_for(run + suffix)
            if v:
                # blank share comes from the run's own manifest, not from a table here --
                # `balanced` was silently identical to `all` once because a hand-written
                # number said otherwise.
                rj = d / "run.json"
                share = (json.loads(rj.read_text()).get("blank_share")
                         if rj.exists() else None)
                mix.append(([label, f"{share:.0%}" if share is not None else "—", lr], v))
                break
    mix_block = render(mix, ["mix", "blank share", "lr"])

    if a.dump:
        a.dump.write_text(grid_block + "\n<!--split-->\n" + mix_block + "\n")
        print(f"-> {a.dump}")
    if a.write:
        p = ROOT / "README.md"
        t = p.read_text()
        t = splice(t, "grid-table", grid_block)
        t = splice(t, "mix-table", mix_block)
        p.write_text(t)
        print(f"-> README.md ({len(grid)} grid rows, {len(mix)} mix rows)")
    print(grid_block)
    print()
    print(mix_block)


if __name__ == "__main__":
    main()
