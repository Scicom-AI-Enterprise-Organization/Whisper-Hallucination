#!/usr/bin/env python3
"""Split the wild arm into train and test, by SOURCE RECORDING.

The arm shipped as `test` only. Training on any of it would contaminate the wild baselines,
so the split has to exist before a single clip is used for training.

Three rules, each because breaking it invalidates something:

1. **Group by recording, never by clip.** Seventeen AudioSet clips can come from one YouTube
   video and a dozen HALAS segments from one earnings call; splitting clips would put near
   neighbours on both sides and the test score would be inflated. The key is the video id for
   AudioSet, the call's `file_id` for HALAS, the meeting for AMI.

2. **HALAS stays entirely in test.** It is the only part of the arm with human span
   annotations across nine ASR systems -- the flagship wild evaluation. Spending it on
   training would buy a little data and cost the benchmark.

3. **`loop` clips stay in test.** They have no reference transcript, so there is no target to
   train toward; they are only meaningful as evaluation.

What remains trainable is the mined `blank_speech` material: real audio a VAD confirms has no
speech, whose correct target is the empty string. That is the same lesson the synthetic blank
arms teach, on audio that actually occurs.

    python scripts/split_wild.py --out audio_wild/split_report.json
"""
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path


def recording_key(collection: str, row: dict) -> str:
    """The unit that must not straddle the split."""
    sid = (row.get("source_id") or row.get("id") or "").strip()
    if collection == "halas":
        # `{segment_id}_{file_id}.wav` -- the call is the file_id.
        stem = sid.rsplit(".", 1)[0]
        return f"halas:{stem.split('_', 1)[1] if '_' in stem else stem}"
    if collection == "audioset":
        return f"audioset:{Path(sid).stem}"          # the YouTube video id
    if collection == "ami":
        # `train_ami_en2001a_h03_mee067_0405743_0405802.wav` -- the meeting is the third
        # field. Taking the first field yields the literal "train" for every clip, which
        # collapses all 235 into one "recording" and sends the lot to whichever side it lands
        # on. A key that groups everything is not a key.
        parts = Path(sid).stem.split("_")
        return f"ami:{parts[2] if len(parts) > 2 else Path(sid).stem}"
    return f"{collection}:{sid or row.get('id')}"


def bucket(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}\x1f{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wild", type=Path, default=Path("audio_wild"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--salt", default="whisper-halluc-wild-v1")
    ap.add_argument("--test-only-collections", nargs="*", default=["halas"])
    ap.add_argument("--test-only-reasons", nargs="*", default=["loop"])
    ap.add_argument("--out", type=Path, default=Path("audio_wild/split_report.json"))
    args = ap.parse_args()

    rows, reasons_forced, coll_forced = [], 0, 0
    for man in sorted(args.wild.rglob("manifest.csv")):
        collection = man.parent.parent.name
        with man.open(encoding="utf-8") as fh:
            fields = None
            for r in csv.DictReader(fh):
                fields = fields or list(r)
                r["_collection"] = collection
                r["_key"] = recording_key(collection, r)
                rows.append((man, r))
    if not rows:
        raise SystemExit("no wild manifests found")

    # Forcing has to apply to the RECORDING, not the clip. A first pass that forced only the
    # `loop` clip of a recording left its sibling `blank_speech` clip in train, which is the
    # leak this function exists to prevent -- the guard below caught it.
    forced = set()
    for _, r in rows:
        if r["_collection"] in args.test_only_collections:
            forced.add(r["_key"]); coll_forced += 1
        elif any(x in (r.get("reasons") or "").split("|") for x in args.test_only_reasons):
            forced.add(r["_key"]); reasons_forced += 1

    assign = {}
    for _, r in rows:
        key = r["_key"]
        if key in assign:
            continue
        assign[key] = "test" if (key in forced or bucket(key, args.salt) < args.test_frac) \
            else "train"

    by_man = defaultdict(list)
    for man, r in rows:
        r["split"] = assign[r["_key"]]
        by_man[man].append(r)

    for man, items in by_man.items():
        fields = [k for k in items[0] if not k.startswith("_")]
        with man.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(items)

    all_rows = [r for _, r in rows]
    train_keys = {r["_key"] for r in all_rows if r["split"] == "train"}
    test_keys = {r["_key"] for r in all_rows if r["split"] == "test"}
    overlap = train_keys & test_keys
    report = {
        "salt": args.salt, "test_frac": args.test_frac,
        "clips": dict(Counter(r["split"] for r in all_rows)),
        "recordings": {"train": len(train_keys), "test": len(test_keys)},
        "recording_overlap": len(overlap),
        "forced_to_test": {"collections": coll_forced, "reasons": reasons_forced},
        "by_collection": {c: dict(Counter(r["split"] for r in all_rows if r["_collection"] == c))
                          for c in sorted({r["_collection"] for r in all_rows})},
        "trainable_blank_speech": sum(
            1 for r in all_rows
            if r["split"] == "train" and "blank_speech" in (r.get("reasons") or "")),
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if overlap:
        raise SystemExit(f"{len(overlap)} recordings straddle the split - refusing")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
