#!/usr/bin/env python3
"""Merge the published lexicons, our mined hallucinations and the translated positives.

The three are evidence of different things and the merge has to keep that straight:

  observed    upstream lists (AGH, Granary, hf_noise) -- phrases Whisper was seen to invent
  mined       our own checkpoints on our own non-speech arms -- same kind of evidence,
              attested against audio we control, with model and arm recorded
  translated  English hallucination phrases rendered into other languages -- these are
              things people SAY, not things Whisper was seen to invent in that language

`provenance` carries that distinction into every downstream manifest. Anything deciding what
is safe to blocklist must read only `observed` and `mined`; `translated` rows are positives
and using them as hallucination evidence would be circular.

Counts are not comparable across provenances either: a translated row inherits the English
phrase's count purely so priority ordering still works, and is marked `count_basis=inherited`.

    python scripts/merge_lexicons.py --out lexicon/expanded_lexicon.csv
"""
import argparse, csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined", type=Path, default=ROOT / "lexicon" / "combined_lexicon.csv")
    ap.add_argument("--mined", type=Path, default=ROOT / "lexicon" / "mined_lexicon.csv")
    ap.add_argument("--translated", type=Path, default=ROOT / "lexicon" / "translated_lexicon.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "lexicon" / "expanded_lexicon.csv")
    args = ap.parse_args()

    merged = {}

    def add(path, provenance, basis):
        if not path.exists():
            print(f"[skip] {path.name} not present")
            return 0
        n = 0
        for r in csv.DictReader(path.open(encoding="utf-8")):
            key = (r["phrase_norm"], r["lang"])
            count = int(r.get("count") or 0)
            cur = merged.get(key)
            if cur is None:
                merged[key] = {"phrase_norm": r["phrase_norm"], "phrase": r["phrase"],
                               "lang": r["lang"], "count": count,
                               "source": r.get("source", provenance),
                               "provenance": provenance, "count_basis": basis}
            else:
                # Same phrase from two provenances: keep the strongest evidence and the
                # larger count, and record both sources.
                cur["count"] = max(cur["count"], count)
                srcs = {cur["source"], r.get("source", provenance)}
                cur["source"] = "|".join(sorted(s for s in srcs if s))
                rank = {"observed": 0, "mined": 1, "translated": 2}
                if rank[provenance] < rank[cur["provenance"]]:
                    cur["provenance"], cur["count_basis"] = provenance, basis
            n += 1
        return n

    n_comb = add(args.combined, "observed", "observed")
    n_mine = add(args.mined, "mined", "observed")
    n_tran = add(args.translated, "translated", "inherited")

    rows = sorted(merged.values(), key=lambda r: -r["count"])
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["phrase_norm", "phrase", "lang", "count",
                                           "source", "provenance", "count_basis"])
        w.writeheader(); w.writerows(rows)

    prov = Counter(r["provenance"] for r in rows)
    langs = Counter(r["lang"] for r in rows)
    ge2 = [r for r in rows if r["count"] >= 2]
    print(f"inputs: combined={n_comb} mined={n_mine} translated={n_tran}")
    print(f"merged: {len(rows)} rows, {len(langs)} languages")
    print(f"  by provenance {dict(prov)}")
    print(f"  count>=2: {len(ge2)} rows across {len({r['lang'] for r in ge2})} languages")
    non_en = [r for r in ge2 if r["lang"] != "en"]
    print(f"  count>=2 non-English: {len(non_en)} rows across {len({r['lang'] for r in non_en})} languages")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
