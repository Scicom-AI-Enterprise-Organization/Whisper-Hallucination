#!/usr/bin/env python3
"""Turn the 40,891-row lexicon into a synthesis queue: filtered, deduplicated, prioritised.

Three decisions, all of which matter at this scale:

  * **Filtered.** 960 entries are degenerate (`त र`, a single token repeated) — Whisper loop
    artefacts, not phrases anyone says. Another 355 run past 200 characters, up to 1,830;
    those are runaway transcripts, not lexicon entries, and they dominate generation cost.
  * **Deduplicated** on (phrase_norm, lang): 40,891 rows hold 40,156 distinct norms.
  * **Prioritised by observed count, descending.** The queue is ~39k clips and many GPU-hours,
    so it has to degrade gracefully: stopping early should leave the most-hallucinated
    phrases done, not a random sample. `абонирайте се` at 2.3M observations is worth more
    than the tail of one-offs.

    python tts/build_lexicon_queue.py --out tts/lexicon_queue.jsonl
"""
import argparse, csv, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def degenerate(phrase: str) -> bool:
    toks = phrase.split()
    return (len(toks) > 1 and len(set(toks)) == 1) or len(set(phrase.replace(" ", ""))) <= 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=ROOT / "lexicon" / "combined_lexicon.csv")
    ap.add_argument("--plan", type=Path, default=ROOT / "tts" / "lexicon_voice_plan.json")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "lexicon_queue.jsonl")
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--per-lang-cap", type=int, default=0,
                    help="keep only the N most-observed phrases per language (0 = all)")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    per_lang = plan["per_language"]
    default_engine = plan["default_for_unmeasured_languages"]["engine"]

    rows = list(csv.DictReader(args.lexicon.open(encoding="utf-8")))
    seen, kept, drops = set(), [], Counter()

    for r in rows:
        phrase, lang = r["phrase"].strip(), r["lang"]
        if not phrase:
            drops["empty"] += 1; continue
        if degenerate(phrase):
            drops["degenerate"] += 1; continue
        if len(phrase) > args.max_chars or len(phrase.split()) > args.max_words:
            drops["too_long"] += 1; continue
        key = (r["phrase_norm"], lang)
        if key in seen:
            drops["duplicate"] += 1; continue
        seen.add(key)
        spec = per_lang.get(lang)
        kept.append({
            "phrase": phrase, "lang": lang,
            "count": int(r.get("count") or 0),
            "source": r.get("source", ""),
            "engine": spec["engine"] if spec else default_engine,
            "voice": (spec or {}).get("voice"),
        })

    kept.sort(key=lambda x: -x["count"])

    if args.per_lang_cap:
        per, capped = Counter(), []
        for item in kept:
            if per[item["lang"]] < args.per_lang_cap:
                per[item["lang"]] += 1
                capped.append(item)
        drops["over_per_lang_cap"] = len(kept) - len(capped)
        kept = capped

    with args.out.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(kept):
            fh.write(json.dumps({"idx": i, **item}, ensure_ascii=False) + "\n")

    by_engine = Counter(x["engine"] for x in kept)
    by_lang = Counter(x["lang"] for x in kept)
    print(f"lexicon rows      {len(rows):>7}")
    for k, v in drops.most_common():
        print(f"  dropped {k:<16} {v:>7}")
    print(f"queued            {len(kept):>7}   ({len(by_lang)} languages)")
    print(f"  by engine       {dict(by_engine)}")
    print(f"  largest langs   {by_lang.most_common(6)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
