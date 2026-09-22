#!/usr/bin/env python3
"""Queue a second voice for every phrase whose language came out of OmniVoice as one speaker.

OmniVoice takes no speaker argument: `tts/voice_diversity.py` measured 33 of its 74 languages
at a median pairwise cosine of 0.75 or worse (0.605 = different speakers, 0.853 = the same
one), i.e. one voice wearing thousands of filenames. The probe
(`tts/build_diversity_probe.py`, 4 languages x 24 phrases x 4 voices) compared the two fixes
on the same phrases:

                yield mk/gu/it        different-voice cosine
  named         86% / 80% / 79%       0.652 / 0.638 / 0.578     <- keeps yield, separates
  cloned        46% / 68% / 53%       0.684 / 0.722 / 0.687     <- pays the +0.339 CER twice

So the top-up runs Multilingual-Expressive in speaker-name mode, which also turned out to
work in languages its TTS ablation never covered (mk, gu). The auto-mode clips are KEPT --
this adds voices to a phrase rather than replacing it.

Languages whose v3 yield was already near zero are skipped: `si` accepts 0.5% of auto-mode
clips because FLEURS has no Sinhala config, so there is no judge floor and the flat gate
rejects everything. That is a judge-coverage problem, and re-voicing it would burn GPU on
clips the filter cannot score either way.

    python tts/build_diversity_topup.py --voices 3            # writes the queue
    python tts/build_diversity_topup.py --dry-run             # just size the job
"""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pick_voices(phrase: str, lang: str, pool: list[str], k: int) -> list[str]:
    """Deterministic, phrase-dependent draw -- the same convention build_lexicon_queue.py uses,
    so a rebuild reproduces the assignment and no voice is over-represented per language."""
    h = int(hashlib.sha1(f"{lang}|{phrase}".encode()).hexdigest()[:8], 16)
    return [pool[(h + i) % len(pool)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diversity", type=Path, default=ROOT / "tts" / "voice_diversity.json")
    ap.add_argument("--corpus", type=Path, default=Path("audio/lexicon_synth_v3"))
    ap.add_argument("--filter-report", type=Path,
                    default=Path("audio/lexicon_synth_v3_omonly/filter_report.json"))
    ap.add_argument("--speakers", type=Path, default=ROOT / "tts" / "expressive_speakers.json")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "diversity_topup_queue.jsonl")
    ap.add_argument("--collapsed-at", type=float, default=0.75)
    ap.add_argument("--min-yield", type=float, default=0.15,
                    help="skip languages whose auto-mode clips the judge already rejects")
    ap.add_argument("--voices", type=int, default=3)
    ap.add_argument("--voice-prefix", default="multilingual-tts_audio_",
                    help="speaker-name family to draw from; these are the names the TTS "
                         "fine-tune was trained on")
    ap.add_argument("--extra-voices", nargs="+", default=["DisfluencySpeech"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    known = json.loads(args.speakers.read_text())["all_speakers"]
    pool = sorted(v for v in known if v.startswith(args.voice_prefix))
    pool += [v for v in args.extra_voices if v in known]
    if len(pool) < args.voices:
        raise SystemExit(f"only {len(pool)} verified speakers match {args.voice_prefix!r}")

    div = json.loads(args.diversity.read_text())["engines"]["omnivoice"]["langs"]
    collapsed = {l for l, v in div.items() if v["median_cosine"] >= args.collapsed_at}
    report = json.loads(args.filter_report.read_text()) if args.filter_report.exists() else {}

    rows = [r for r in csv.DictReader(
        (args.corpus / "omnivoice" / "manifest.shard0.csv").open(encoding="utf-8"))
        if r["status"] == "ok" and r["lang"] in collapsed]

    seen, items, skipped, per_lang = set(), [], {}, Counter()
    for r in rows:
        lang = r["lang"]
        y = (report.get(lang) or {}).get("yield")
        if y is not None and y < args.min_yield:
            skipped[lang] = y
            continue
        key = (lang, r["phrase"])
        if key in seen:
            continue
        seen.add(key)
        for v in pick_voices(r["phrase"], lang, pool, args.voices):
            items.append({"idx": len(items), "phrase": r["phrase"], "lang": lang,
                          "count": int(r["count"]), "source": "diversity_topup",
                          "engine": "multilingual-expressive", "voice": v})
        per_lang[lang] += 1

    print(f"speaker pool        {len(pool)} verified names")
    print(f"collapsed languages {len(collapsed)} at >= {args.collapsed_at}")
    print(f"skipped (yield < {args.min_yield:.0%})  "
          f"{', '.join(f'{l} {y:.1%}' for l, y in sorted(skipped.items())) or 'none'}")
    print(f"phrases to re-voice {len(seen):,} over {len(per_lang)} languages")
    print(f"clips queued        {len(items):,}  ({args.voices} voices each)")
    print("largest languages   " + ", ".join(f"{l}:{n}" for l, n in per_lang.most_common(10)))
    if args.dry_run:
        return
    with args.out.open("w", encoding="utf-8") as f:
        for e in items:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
