#!/usr/bin/env python3
"""Re-aggregate tts_scores.json with the same phrase filter the VC scorer uses.

`score_tts.py` averages every clip, degenerate lexicon entries included -- and those entries
(`त र`, `સ સ સ સ સ સ સ`) score CER in the tens for every system, which is how `bn` ended up
at 22.9 and why the TTS means in CLAUDE.md never matched the JSON. Excluding them here makes
the TTS and VC tables comparable, and costs nothing: the per-clip CERs are already stored, so
no ASR is re-run.

    python tts/aggregate_tts.py
    python tts/aggregate_tts.py --valid-speakers-only     # drop the unverified-name clips

`--valid-speakers-only` drops every phrase whose Multilingual-Expressive clip was conditioned
on a speaker name absent from `tts/expressive_speakers.json` -- the same `Rahman` mistake that
invalidated the VC named-mode row. An unknown name does not raise, it conditions on a token the
fine-tune never saw, so those clips measure nothing. The phrases are dropped for EVERY system,
not just ours, or the systems would no longer be compared on the same text.
"""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def degenerate(phrase: str) -> str:
    """Whisper loop artefacts captured as lexicon phrases — see score_vc.py."""
    toks = phrase.split()
    if len(toks) > 1 and len(set(toks)) == 1:
        return "single repeated token"
    if len(set(phrase.replace(" ", ""))) <= 2:
        return "<=2 distinct characters"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=ROOT / "tts" / "tts_scores.json")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "tts_scores_summary.json")
    ap.add_argument("--speakers", type=Path, default=ROOT / "tts" / "expressive_speakers.json")
    ap.add_argument("--valid-speakers-only", action="store_true",
                    help="drop phrases voiced by a speaker name absent from the inventory")
    args = ap.parse_args()

    raw = json.loads(args.scores.read_text())

    dropped = set()
    if args.valid_speakers_only:
        known = set(json.loads(args.speakers.read_text())["all_speakers"])
        for r in raw.values():
            for d in r["details"]:
                spk = d.get("speaker")
                if spk and spk not in known:
                    dropped.add((d["lang"], d["phrase"]))
        if not dropped:
            print("every speaker name is in the inventory -- nothing dropped")
    out, per_lang_all = {}, defaultdict(dict)

    for name, r in raw.items():
        kept = [d for d in r["details"]
                if not degenerate(d["phrase"]) and (d["lang"], d["phrase"]) not in dropped]
        by_lang = defaultdict(list)
        for d in kept:
            by_lang[d["lang"]].append(d["cer"])
        cers = [d["cer"] for d in kept]
        out[name] = {
            "n": len(kept), "n_degenerate_excluded": len(r["details"]) - len(kept),
            "synth_ok": sum(d.get("status") == "ok" for d in kept),
            "mean_cer": round(statistics.mean(cers), 4),
            "median_cer": round(statistics.median(cers), 4),
            "per_lang_cer": {k: round(statistics.mean(v), 4) for k, v in sorted(by_lang.items())},
        }
        for lang, v in out[name]["per_lang_cer"].items():
            per_lang_all[lang][name] = v

    # wins: one per language, ties broken by the systems' order — the same convention
    # score_tts.py prints, so the counts still sum to the language count.
    wins = defaultdict(int)
    order = list(raw)
    for lang, vals in per_lang_all.items():
        best = min(vals.values())
        winner = next(n for n in order if vals.get(n) == best)
        wins[winner] += 1
    for name in out:
        out[name]["wins"] = wins[name]

    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    hdr = f"{'system':<12}{'n':>5}{'mean':>9}{'median':>9}{'wins':>6}{'langs':>7}"
    print(hdr); print("-" * len(hdr))
    for name, r in sorted(out.items(), key=lambda kv: kv[1]["mean_cer"]):
        print(f"{name:<12}{r['n']:>5}{r['mean_cer']:>9.3f}{r['median_cer']:>9.3f}"
              f"{r['wins']:>6}{len(r['per_lang_cer']):>7}")
    print(f"\nexcluded {out[list(out)[0]]['n_degenerate_excluded']} degenerate phrases per system")
    if dropped:
        print(f"excluded {len(dropped)} phrases voiced by an unverified speaker name")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
