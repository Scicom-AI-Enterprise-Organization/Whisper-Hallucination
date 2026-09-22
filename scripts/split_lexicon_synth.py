#!/usr/bin/env python3
"""Split the synthesised lexicon into train and test, by PHRASE, with the benchmark protected.

Three rules, each of which exists because breaking it invalidates a number we would later
report:

1. **Split on (phrase_norm, lang), never on clips.** The same phrase is rendered in several
   voices; splitting clips would put `terima kasih` in both halves and a model would be
   evaluated on text it trained on. Every clip of a phrase goes to the same side.

2. **Benchmark phrases go to TEST, never TRAIN.** `phrases/targets.csv` drives the published
   `genuine_isolated` arm, and the whole repo rests on "never train on the benchmark"
   (CLAUDE.md). Synthetic audio of a benchmark phrase is still that phrase, so training on it
   contaminates the benchmark just as surely as training on its audio would.

3. **Deterministic and stratified.** The assignment is a hash of (phrase_norm, lang), so it
   is reproducible without storing a map and stable when the corpus grows; stratifying by
   language keeps small languages present in both halves instead of landing entirely in one.

    python scripts/split_lexicon_synth.py --accepted audio/lexicon_synth_v2/accepted.csv
"""
import argparse, csv, hashlib, json, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "").strip().lower())


def bucket(phrase_norm: str, lang: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}\x1f{lang}\x1f{phrase_norm}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accepted", type=Path, nargs="+",
                    default=[ROOT / "audio" / "lexicon_synth_v2" / "accepted.csv"])
    ap.add_argument("--audio-root", type=Path, nargs="*", default=None,
                    help="where each --accepted file's clips actually live, in the same order. "
                         "Defaults to each file's own directory, which is WRONG whenever the "
                         "filter wrote its report somewhere other than the synthesis root -- "
                         "the v3 run filtered into lexicon_synth_v3_{sconly,omonly} while the "
                         "audio stayed in lexicon_synth_v3.")
    ap.add_argument("--targets", type=Path, default=ROOT / "phrases" / "targets.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "audio" / "lexicon_synth_v2")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--salt", default="whisper-halluc-lexicon-v1",
                    help="change only to reshuffle deliberately; the split is a hash, not a file")
    ap.add_argument("--min-lang-clips", type=int, default=4,
                    help="languages below this go entirely to train; a 1-clip test set is noise")
    args = ap.parse_args()

    protected = set()
    if args.targets.exists():
        for r in csv.DictReader(args.targets.open(encoding="utf-8")):
            protected.add((norm(r["phrase"]), r.get("lang", "")))
            protected.add((norm(r["phrase"]), ""))        # match regardless of language tag
    print(f"benchmark phrases protected from train: {len(protected)//2}")

    roots = args.audio_root or [p.parent for p in args.accepted]
    if len(roots) != len(args.accepted):
        raise SystemExit(f"--audio-root has {len(roots)} entries for "
                         f"{len(args.accepted)} --accepted files")

    rows = []
    for path, audio_root in zip(args.accepted, roots):
        for r in csv.DictReader(path.open(encoding="utf-8")):
            # Which corpus root the clip's audio lives under. Kept as a real column (not an
            # underscore-prefixed scratch field) because the release builder needs it: clips
            # now come from more than one synthesis root -- the original run plus the
            # speaker-diversity top-up -- and `audio_filepath` is relative to its own root.
            r["corpus_root"] = str(audio_root)
            rows.append(r)
    print(f"{len(rows)} accepted clips from {len(args.accepted)} manifest(s)")
    for root, n in Counter(r["corpus_root"] for r in rows).most_common():
        print(f"    {n:>7} from {root}")

    # Group clips by phrase: the unit of assignment is the phrase, not the clip.
    by_phrase = defaultdict(list)
    for r in rows:
        by_phrase[(norm(r["phrase"]), r["lang"])].append(r)
    per_lang_clips = Counter(r["lang"] for r in rows)

    assign, reasons = {}, Counter()
    for key in sorted(by_phrase):
        pn, lang = key
        if (pn, lang) in protected or (pn, "") in protected:
            assign[key] = "test"; reasons["benchmark_phrase"] += 1
        elif per_lang_clips[lang] < args.min_lang_clips:
            assign[key] = "train"; reasons["language_too_small_for_test"] += 1
        else:
            assign[key] = "test" if bucket(pn, lang, args.salt) < args.test_frac else "train"
            reasons["hashed"] += 1

    out_rows = defaultdict(list)
    for key, clips in by_phrase.items():
        for r in clips:
            r = {k: v for k, v in r.items() if not k.startswith("_")}
            r["split"] = assign[key]
            out_rows[assign[key]].append(r)

    # Roots can disagree on columns if one was filtered by an older version; take the union
    # so no row silently loses a field on write.
    fields, seen_f = [], set()
    for items in out_rows.values():
        for r in items:
            for k in r:
                if k not in seen_f:
                    seen_f.add(k); fields.append(k)
    for split, items in out_rows.items():
        path = args.out_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader(); w.writerows(items)
        print(f"  {split}: {len(items)} clips -> {path.name}")

    # ---- verification: the split is only worth anything if these hold ------------------
    train_keys = {(norm(r["phrase"]), r["lang"]) for r in out_rows["train"]}
    test_keys = {(norm(r["phrase"]), r["lang"]) for r in out_rows["test"]}
    overlap = train_keys & test_keys
    leaked = {k for k in train_keys if k in protected or (k[0], "") in protected}

    report = {
        "salt": args.salt, "test_frac": args.test_frac,
        "clips": {k: len(v) for k, v in out_rows.items()},
        "phrases": {"train": len(train_keys), "test": len(test_keys)},
        "phrase_overlap_train_test": len(overlap),
        "benchmark_phrases_in_train": len(leaked),
        "assignment_reasons": dict(reasons),
        "clips_by_corpus_root": {k: v for k, v in
                                 Counter(r["corpus_root"] for items in out_rows.values()
                                         for r in items).most_common()},
        "languages": {
            "train": len({r["lang"] for r in out_rows["train"]}),
            "test": len({r["lang"] for r in out_rows["test"]}),
            "in_both": len({r["lang"] for r in out_rows["train"]} & {r["lang"] for r in out_rows["test"]}),
        },
    }
    (args.out_dir / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nphrases  train={len(train_keys)}  test={len(test_keys)}")
    print(f"languages train={report['languages']['train']} test={report['languages']['test']} "
          f"both={report['languages']['in_both']}")
    print(f"VERIFY phrase overlap train/test : {len(overlap)}  {'OK' if not overlap else 'FAIL'}")
    print(f"VERIFY benchmark phrases in train: {len(leaked)}  {'OK' if not leaked else 'FAIL'}")
    if overlap or leaked:
        raise SystemExit("split is unsafe - refusing to leave it in place")
    print(f"-> {args.out_dir}/train.csv, test.csv, split_report.json")


if __name__ == "__main__":
    main()
