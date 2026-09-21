#!/usr/bin/env python3
"""Harvest new lexicon entries from what our own benchmark runs actually produced.

The published lexicons were built exactly this way: run an ASR on audio with no speech and
write down what it invents. We now have five checkpoints x three non-speech arms x ~1,810
clips of our own, so the outputs are sitting on disk -- and unlike the upstream lists they
are attested against audio we control, with the model and arm recorded.

This is the OPPOSITE artefact from `translate_lexicon.py`: these are observed hallucinations
(evidence about what the model invents), not translated positives (evidence about what people
say). Rows carry `source=mined:<models>` so the two never get conflated downstream.

Language is detected from the text, because the result records do not store Whisper's
detected language. Detection is only used to route synthesis, so a wrong guess costs a clip,
not a claim.

    .venv_bench/bin/python scripts/mine_bench_hallucinations.py --min-count 2
"""
import argparse, csv, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("music", "nonspeech", "silence")

# Script ranges are decisive where they apply; Latin needs a real detector.
SCRIPT_LANG = [
    (r"[぀-ゟ゠-ヿ]", "ja"), (r"[가-힯]", "ko"),
    (r"[一-鿿]", "zh"),              (r"[฀-๿]", "th"),
    (r"[؀-ۿ]", "ar"),              (r"[֐-׿]", "he"),
    (r"[ऀ-ॿ]", "hi"),              (r"[஀-௿]", "ta"),
    (r"[ঀ-৿]", "bn"),              (r"[ఀ-౿]", "te"),
    (r"[ಀ-೿]", "kn"),              (r"[ഀ-ൿ]", "ml"),
    (r"[઀-૿]", "gu"),              (r"[਀-੿]", "pa"),
    (r"[ក-៿]", "km"),              (r"[຀-໿]", "lo"),
    (r"[က-႟]", "my"),              (r"[Ⴀ-ჿ]", "ka"),
    (r"[԰-֏]", "hy"),              (r"[Ѐ-ӿ]", "ru"),
    (r"[Ͱ-Ͽ]", "el"),              (r"[ሀ-፿]", "am"),
]


def degenerate(phrase: str) -> bool:
    toks = phrase.split()
    return (len(toks) > 1 and len(set(toks)) == 1) or len(set(phrase.replace(" ", ""))) <= 2


def detect_lang(text: str, fallback="en") -> str:
    for pat, lang in SCRIPT_LANG:
        if re.search(pat, text):
            return lang
    try:
        import langid
        return langid.classify(text)[0]
    except Exception:
        return fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "bench" / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "lexicon" / "mined_lexicon.csv")
    ap.add_argument("--min-count", type=int, default=2,
                    help="how many (model, clip) outputs must agree before it is a phrase")
    ap.add_argument("--max-words", type=int, default=40)
    args = ap.parse_args()

    hits = defaultdict(lambda: {"count": 0, "models": set(), "arms": set()})
    scanned = 0
    for mdir in sorted(p for p in args.results.glob("*") if p.is_dir()):
        for arm in ARMS:
            f = mdir / f"{arm}.jsonl"
            if not f.exists():
                continue
            for line in f.open(encoding="utf-8"):
                scanned += 1
                hyp = (json.loads(line).get("hyp") or "").strip()
                if not hyp:
                    continue
                key = unicodedata.normalize("NFKC", hyp)
                h = hits[key]
                h["count"] += 1
                h["models"].add(mdir.name)
                h["arms"].add(arm)

    rows, drops = [], Counter()
    for phrase, meta in hits.items():
        if meta["count"] < args.min_count:
            drops["below_min_count"] += 1; continue
        if degenerate(phrase):
            drops["degenerate"] += 1; continue
        if len(phrase.split()) > args.max_words:
            drops["too_long"] += 1; continue
        if len(phrase.strip()) < 2:
            drops["too_short"] += 1; continue
        rows.append({
            "phrase_norm": phrase.lower(), "phrase": phrase,
            "lang": detect_lang(phrase), "count": meta["count"],
            "source": "mined:" + "|".join(sorted(m.split("-")[-1] for m in meta["models"])),
            "arms": "|".join(sorted(meta["arms"])),
            "n_models": len(meta["models"]),
        })
    rows.sort(key=lambda r: -r["count"])

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["phrase_norm", "phrase", "lang", "count",
                                           "source", "arms", "n_models"])
        w.writeheader(); w.writerows(rows)

    by_lang = Counter(r["lang"] for r in rows)
    multi = sum(1 for r in rows if r["n_models"] >= 2)
    print(f"scanned {scanned} non-speech hypotheses; {len(hits)} distinct outputs")
    for k, v in drops.most_common():
        print(f"  dropped {k:<18} {v}")
    print(f"kept {len(rows)} phrases across {len(by_lang)} languages "
          f"({multi} seen by 2+ checkpoints)")
    print("  languages:", by_lang.most_common(12))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
