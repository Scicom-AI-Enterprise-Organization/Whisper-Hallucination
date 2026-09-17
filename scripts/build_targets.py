#!/usr/bin/env python3
"""Build phrases/targets.csv -- the high-risk phrases the contrastive arm is built around.

A "high-risk" phrase is one Whisper emits on non-speech AND that people genuinely say.
That overlap is the whole problem: you cannot filter `terima kasih` out of a transcript
on text alone without destroying every real `terima kasih`. Each target therefore needs
both a hallucinated and a genuine audio pool.

Selection:
  * top-N observed phrases per language from lexicon/combined_lexicon.csv
  * dropped if it is a bare function word (`so`, `the`, `you`) -- those are noise in the
    tally, not phrases anyone would filter on
  * plus a hand-curated Malaysian set (ms/zh/ta), which public lexicons barely cover:
    combined_lexicon.csv has only 6 `ms` phrases, and Scicom's traffic is ms/en/zh/ta
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _disp(p: Path) -> str:
    """Path for display; falls back to the absolute path when it is outside ROOT."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


# Phrases that are pure decoder filler rather than something a filter would target.
STOPWORDS = {
    "so", "the", "you", "oh", "a", "i", "and", "it", "uh", "um", "yeah", "yes", "no",
    "okay", "ok", "hmm", "mm", "well", "but", "to", "of", "is", "that", "this",
}

# Hand-curated. Public lexicons under-cover Malaysian languages, and these are exactly
# the phrases a Malaysian call-centre agent says all day -- so the genuine pool is large
# and the confusion is operationally expensive.
CURATED = [
    # (phrase, lang, note)
    ("terima kasih",                    "ms", "canonical ms hallucination AND the most common call-centre closer"),
    ("terima kasih kerana menonton",    "ms", "YouTube-outro artefact; near-zero genuine rate"),
    ("terima kasih banyak banyak",      "ms", "intensified form"),
    ("sama-sama",                       "ms", "reply pair to terima kasih"),
    ("selamat sejahtera",               "ms", "greeting, high genuine rate"),
    ("selamat tinggal",                 "ms", "farewell"),
    ("sila tunggu sebentar",            "ms", "hold phrase, very high genuine rate in call centre"),
    ("baik encik",                      "ms", "agent backchannel"),
    ("jangan lupa subscribe",           "ms", "YouTube artefact; near-zero genuine rate"),
    ("thank you",                       "en", "highest-count hallucination overall"),
    ("thank you for calling",           "en", "call-centre closer, high genuine rate"),
    ("thanks for watching",             "en", "YouTube artefact; near-zero genuine rate"),
    ("please subscribe",                "en", "YouTube artefact"),
    ("please hold",                     "en", "call-centre, high genuine rate"),
    ("谢谢",                             "zh", "canonical zh hallucination and a real closer"),
    ("谢谢观看",                          "zh", "YouTube artefact"),
    ("感谢您联系",                        "zh", "call-centre closer"),
    ("请稍等",                            "zh", "hold phrase"),
    ("நன்றி",                            "ta", "ta thank-you; ta arm gates on CER not WER"),
    ("மிக்க நன்றி",                       "ta", "intensified form"),
]


def main() -> int:
    lex = list(csv.DictReader((ROOT / "lexicon" / "combined_lexicon.csv").open(encoding="utf-8")))
    by_lang: dict[str, list[dict]] = {}
    for r in lex:
        by_lang.setdefault(r["lang"], []).append(r)

    rows, seen = [], set()

    def add(phrase, lang, source, count, note):
        key = (phrase.strip().casefold(), lang)
        if key in seen or not phrase.strip():
            return
        seen.add(key)
        rows.append({
            "phrase": phrase.strip(), "lang": lang, "source": source,
            "observed_hallucination_count": count, "note": note,
        })

    for phrase, lang, note in CURATED:
        hit = next((r for r in by_lang.get(lang, []) if r["phrase_norm"] == phrase.casefold()), None)
        add(phrase, lang, "curated", int(hit["count"]) if hit else 0, note)

    for lang, entries in by_lang.items():
        entries.sort(key=lambda r: -int(r["count"]))
        kept = 0
        for r in entries:
            if kept >= 15:
                break
            p = r["phrase_norm"]
            if not p or p in STOPWORDS or len(p) < 3 or int(r["count"]) < 2:
                continue
            add(r["phrase"], lang, "lexicon_top", int(r["count"]), "")
            kept += 1

    rows.sort(key=lambda r: (r["lang"] != "ms", r["lang"], -r["observed_hallucination_count"]))
    out = ROOT / "phrases" / "targets.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    langs = {r["lang"] for r in rows}
    print(f"wrote {len(rows)} target phrases across {len(langs)} languages -> {_disp(out)}")
    for lang in ["ms", "en", "zh", "ta"]:
        sub = [r for r in rows if r["lang"] == lang]
        print(f"  {lang}: {len(sub)}  e.g. " + ", ".join(repr(r['phrase']) for r in sub[:4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
