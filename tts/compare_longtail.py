#!/usr/bin/env python3
"""Re-decide engine routing on REAL lexicon phrases, not the short probe set.

`lexicon_voice_plan.json` was built from 2-3 phrases per language, and those were the
high-frequency ones -- `thank you`, `terima kasih`. The actual lexicon medians 9 words, and
the QA on the scaled run shows Multilingual-Expressive degrading badly there (ta 1.49,
el 1.37) with duration ratios past 2x, i.e. it keeps talking. That is the same runaway its
cloning path shows, and the probe could not have caught it.

So: take the phrases the scaled run actually queued, stratified by length, and score both
engines on them.

    .venv_bench/bin/python tts/compare_longtail.py --langs ta el ur pl en --per-lang 8
"""
import argparse, json, statistics, subprocess, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=ROOT / "tts" / "lexicon_queue.jsonl")
    ap.add_argument("--langs", nargs="+", default=["ta", "el", "ur", "pl", "fr", "en"])
    ap.add_argument("--per-lang", type=int, default=8)
    ap.add_argument("--min-words", type=int, default=5, help="the long tail is the problem case")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "longtail_probe_queue.jsonl")
    args = ap.parse_args()

    items = [json.loads(l) for l in args.queue.open(encoding="utf-8")]
    by_lang = defaultdict(list)
    for x in items:
        if x["lang"] in args.langs and len(x["phrase"].split()) >= args.min_words:
            by_lang[x["lang"]].append(x)

    picked = []
    for lang in args.langs:
        pool = by_lang.get(lang, [])
        if not pool:
            print(f"[warn] no phrases for {lang} with >= {args.min_words} words")
            continue
        # spread across the length range rather than taking the head
        pool.sort(key=lambda x: len(x["phrase"].split()))
        step = max(1, len(pool) // args.per_lang)
        picked += pool[::step][:args.per_lang]

    # Same phrases, once per engine, with a distinct idx space so manifests do not collide.
    rows = []
    for engine in ("multilingual-expressive", "omnivoice"):
        for j, x in enumerate(picked):
            rows.append({**x, "idx": (0 if engine == "multilingual-expressive" else 500000) + j,
                         "engine": engine,
                         "voice": x["voice"] or "multilingual-tts_audio_Serena"})
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(x["phrase"].split()) for x in picked]
    print(f"{len(picked)} phrases across {len(set(x['lang'] for x in picked))} languages, "
          f"{statistics.median(lens):.0f} words median (min {min(lens)}, max {max(lens)})")
    print(f"-> {args.out}  ({len(rows)} rows: each phrase once per engine)")


if __name__ == "__main__":
    main()
