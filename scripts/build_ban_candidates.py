#!/usr/bin/env python3
"""Decide which hallucination phrases are SAFE TO BAN, from evidence rather than intuition.

The whole problem with a text blocklist is that `terima kasih` is both the top Malay
hallucination and the top call-centre closer. But not every phrase is like that:
`terima kasih kerana menonton` ("thanks for watching") is a YouTube-outro artefact that
essentially nobody says on a support call. Those are free to drop.

This crosses the 40,891-phrase lexicon against every genuine transcript we hold and
classifies each phrase by whether it ACTUALLY occurs in real speech:

  unsafe              appears in genuine speech -> banning it destroys real transcriptions
  safe_candidate      never appears, multi-word, and frequently hallucinated
  weak_evidence       never appears, but too rare or too short to be confident

A post-filter should run in EXACT mode (drop the output only when the whole transcript is
the phrase), because that is the shape a hallucination takes. Substring counts are reported
too, so the cost of a more aggressive filter is visible rather than guessed.

Output: phrases/ban_candidates.csv
"""
import csv, re, sys, unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _disp(p: Path) -> str:
    """Path for display; falls back to the absolute path when it is outside ROOT."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

_PUNCT = re.compile(r"[.?!,;:\"'`´()\[\]…]+")

GENUINE_ARMS = ["genuine", "genuine_isolated", "librispeech_test_clean"]


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "").strip().casefold()
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", t)).strip()


def main() -> int:
    # --- genuine speech we can actually check against ---
    exact = Counter()          # whole transcript == phrase
    texts, hours = [], 0.0
    for arm in GENUINE_ARMS:
        man = ROOT / "audio" / arm / "manifest.csv"
        if not man.exists():
            print(f"  (skip {arm}: not built)"); continue
        for r in csv.DictReader(man.open(encoding="utf-8")):
            t = norm(r.get("text") or r.get("reference_text") or "")
            if t:
                texts.append(t); exact[t] += 1
            hours += float(r.get("duration_s") or 0) / 3600
    if not texts:
        sys.exit("no genuine transcripts found - build the genuine arms first")
    blob = "\n".join(texts)
    print(f"genuine corpus: {len(texts)} transcripts, {hours:.1f} h")

    lex = list(csv.DictReader((ROOT / "lexicon" / "combined_lexicon.csv").open(encoding="utf-8")))
    rows = []
    for r in lex:
        p, cnt = r["phrase_norm"], int(r["count"])
        if not p:
            continue
        nw = len(p.split())
        e = exact.get(p, 0)
        # Substring search only for phrases long enough that a hit means something;
        # scanning for "so" or "the" would match everything and tell us nothing.
        sub = blob.count(p) if nw >= 2 and len(p) >= 6 else -1

        if e > 0 or sub > 0:
            verdict, why = "unsafe", f"occurs in genuine speech (exact={e}, substring={max(sub,0)})"
        elif nw >= 3 and cnt >= 20:
            verdict, why = "safe_candidate", "multi-word, frequently hallucinated, never observed genuine"
        elif nw >= 2 and cnt >= 100:
            verdict, why = "safe_candidate", "two-word, heavily hallucinated, never observed genuine"
        else:
            verdict, why = "weak_evidence", f"never observed genuine, but only {nw} word(s) / count {cnt}"

        rows.append({
            "phrase": r["phrase"], "phrase_norm": p, "lang": r["lang"],
            "hallucination_count": cnt, "genuine_exact": e,
            "genuine_substring": sub if sub >= 0 else "",
            "n_words": nw, "verdict": verdict, "reason": why,
        })

    rows.sort(key=lambda x: ({"safe_candidate": 0, "unsafe": 1, "weak_evidence": 2}[x["verdict"]],
                             -x["hallucination_count"]))
    out = ROOT / "phrases" / "ban_candidates.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    v = Counter(x["verdict"] for x in rows)
    print(f"\n{len(rows)} phrases -> {_disp(out)}")
    for k in ("safe_candidate", "unsafe", "weak_evidence"):
        print(f"  {k:<16} {v[k]:>6}")
    print("\ntop safe_candidate (free to drop):")
    for x in [r for r in rows if r["verdict"] == "safe_candidate"][:12]:
        print(f"  {x['hallucination_count']:>7}  [{x['lang']}]  {x['phrase'][:58]}")
    print("\nunsafe, highest hallucination count (banning these costs real transcripts):")
    for x in sorted([r for r in rows if r["verdict"] == "unsafe"],
                    key=lambda r: -r["hallucination_count"])[:10]:
        print(f"  {x['hallucination_count']:>7}  [{x['lang']}]  {x['phrase'][:40]:<40} genuine={x['genuine_exact']}/{x['genuine_substring']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
