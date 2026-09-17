#!/usr/bin/env python3
"""Merge the four vendored hallucination-phrase sources into one normalised lexicon.

Sources (see SOURCES.md for provenance/licence):
  agh_boh.csv                        AGH DSP "Bag of Hallucinations"  (MIT)
  agh_hallucination_list.csv         AGH DSP full tally               (MIT)
  hf_whisper_hallucinations_phrases  sachaarbonel/whisper-hallucinations (MIT)
  granary/<lang>.txt                 NVIDIA NeMo SDP Granary          (Apache-2.0)

Output: lexicon/combined_lexicon.csv  with columns
  phrase_norm, phrase, lang, count, source
"""
import csv, re, sys, unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _disp(p: Path) -> str:
    """Path for display; falls back to the absolute path when it is outside ROOT."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

LEX = ROOT / "lexicon"
OUT = LEX / "combined_lexicon.csv"

# Granary lines are "<phrase> <count>", count is an int or -1 (manually added).
GRANARY_LINE = re.compile(r"^(?P<phrase>.*?)\s+(?P<count>-?\d+)$")


def norm(text: str) -> str:
    """Casefold + strip punctuation/diacritic-preserving NFC + collapse whitespace."""
    text = unicodedata.normalize("NFC", text).strip().casefold()
    text = re.sub(r"[.?!,;:\"'`´()\[\]…]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rows():
    with (LEX / "agh_boh.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield r["prediction"], "en", int(r["number of occurrences in noise"]), "agh_boh"

    with (LEX / "agh_hallucination_list.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield r["prediction"], "en", int(r["number of occurrences in noise"]), "agh_full"

    with (LEX / "hf_whisper_hallucinations_phrases.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield r["phrase"], r["lang"], int(r["count"]), "hf_noise"

    for path in sorted((LEX / "granary").glob("*.txt")):
        lang = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = GRANARY_LINE.match(line)
            phrase, count = (m["phrase"], int(m["count"])) if m else (line, -1)
            yield phrase, lang, count, "granary"


def main() -> int:
    # Key on (normalised phrase, lang) so the same phrase in two languages stays distinct.
    merged: dict[tuple[str, str], dict] = {}
    per_source = defaultdict(int)
    for phrase, lang, count, source in rows():
        key = norm(phrase)
        if not key:
            continue
        per_source[source] += 1
        slot = merged.setdefault(
            (key, lang),
            {"phrase_norm": key, "phrase": phrase.strip(), "lang": lang, "count": 0, "sources": set()},
        )
        # -1 is a manual entry with no observed tally; don't let it drag the sum down.
        slot["count"] += max(count, 0)
        slot["sources"].add(source)

    out = sorted(merged.values(), key=lambda r: (-r["count"], r["phrase_norm"]))
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phrase_norm", "phrase", "lang", "count", "source"])
        for r in out:
            w.writerow([r["phrase_norm"], r["phrase"], r["lang"], r["count"], "|".join(sorted(r["sources"]))])

    langs = {r["lang"] for r in out}
    print(f"read: " + ", ".join(f"{k}={v}" for k, v in sorted(per_source.items())))
    print(f"wrote {len(out)} unique (phrase, lang) rows across {len(langs)} languages -> {_disp(OUT)}")
    print("\ntop 15 by observed count:")
    for r in out[:15]:
        print(f"  {r['count']:>6}  [{r['lang']}]  {r['phrase'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
