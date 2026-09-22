#!/usr/bin/env python3
"""Package the synthesised lexicon as a `lexicon_synth` config with train/test splits.

This is NOT a benchmark arm. The published dataset's eight audio configs are all `test`
because training on them invalidates every number the card reports; this config deliberately
ships a `train` split, so it has to be unmistakably separate from them -- different config
name, its own card section, and the warning kept next to the data.

The two guarantees that make the train split safe are enforced upstream by
`scripts/split_lexicon_synth.py` (no phrase in both splits; no `targets.csv` phrase in train)
and re-verified here before anything is written, because a release is the wrong place to
discover a leak.

Parquet is written with pyarrow directly rather than `datasets`: encoding an Audio column
needs torchcodec, while the on-disk format is only `struct<bytes, path>` plus a `huggingface`
schema-metadata blob (CLAUDE.md). `use_dictionary=[]` because every FLAC value is unique and
already compressed, so a dictionary just stores the payload twice.

    .venv_bench/bin/python scripts/build_lexicon_synth_release.py --out build/lexicon_synth
"""
import argparse, csv, io, json, sys, unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Features, Value

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
_AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
_ARROW = {"string": pa.string(), "int32": pa.int32(), "int64": pa.int64(),
          "float32": pa.float32(), "float64": pa.float64(), "bool": pa.bool_()}
ENGINE_DIR = {"multilingual-expressive": "scicom", "omnivoice": "omnivoice"}


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "").strip().lower())


def flac_bytes(path: Path) -> bytes:
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR}"
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def table(recs, feats: Features) -> pa.Table:
    fields, arrays = [], []
    for name, spec in feats.items():
        if isinstance(spec, Audio):
            fields.append(pa.field(name, _AUDIO_TYPE))
            arrays.append(pa.array([r[name] for r in recs], type=_AUDIO_TYPE))
        else:
            t = _ARROW[spec.dtype]
            fields.append(pa.field(name, t))
            arrays.append(pa.array([r[name] for r in recs], type=t))
    schema = pa.schema(fields, metadata={
        b"huggingface": json.dumps({"info": {"features": feats.to_dict()}}).encode()})
    return pa.Table.from_arrays(arrays, schema=schema)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path, default=ROOT / "audio" / "lexicon_synth_v3")
    ap.add_argument("--targets", type=Path, default=ROOT / "phrases" / "targets.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "lexicon_synth")
    ap.add_argument("--config-name", default="lexicon_synth")
    ap.add_argument("--rows-per-file", type=int, default=4000,
                    help="shard each split so no single parquet approaches the Hub viewer's "
                         "300 MB scan limit; ~36 KB per clip, so 4,000 rows is ~145 MB")
    ap.add_argument("--row-group-size", type=int, default=500,
                    help="rows per row group. The default (a whole file) makes the viewer read "
                         "hundreds of MB to show ten rows; 500 rows is ~18 MB")
    ap.add_argument("--primary-root", type=Path, default=None,
                    help="the corpus root whose clips keep bare ids, because they are already "
                         "published under them. Defaults to --synth, which is only right when "
                         "the split lives beside the audio; once the split is rebuilt into its "
                         "own directory this has to name the original synthesis root.")
    args = ap.parse_args()

    feats = Features({
        "id": Value("string"), "audio": Audio(sampling_rate=SR),
        "phrase": Value("string"), "lang": Value("string"),
        "engine": Value("string"), "voice": Value("string"),
        "observed_count": Value("int64"), "duration_s": Value("float32"),
        "cer": Value("float32"), "asr_hypothesis": Value("string"),
        "split": Value("string"), "meta": Value("string"),
    })

    protected = set()
    if args.targets.exists():
        for r in csv.DictReader(args.targets.open(encoding="utf-8")):
            protected.add(norm(r["phrase"]))

    # `meta` (model, voice, decoding parameters, phrase provenance) is written by the
    # synthesiser but was dropped by an earlier version of the filter, so join it back from
    # the synthesis manifests rather than shipping empty provenance.
    # Clips come from more than one synthesis root now (the original run plus the
    # speaker-diversity top-up), so every lookup is keyed by root as well: `idx` restarts at 0
    # in each root, and so would collide.
    # The primary root keeps bare ids so already-published clips keep the id they shipped
    # with; every other root is tagged. Renaming every row of an existing release to make the
    # ids uniform would be churn for cosmetics.
    primary = str(args.primary_root or args.synth)

    def root_tag(root):
        if root == primary:
            return ""
        name = Path(root).name
        base = Path(primary).name
        return (name[len(base):].strip("_") if name.startswith(base) else name) + "_"

    def roots_in(split_rows):
        return {r.get("corpus_root") or primary for r in split_rows}

    split_rows = {s: list(csv.DictReader((args.synth / f"{s}.csv").open(encoding="utf-8")))
                  for s in ("train", "test")}
    all_roots = sorted(roots_in(split_rows["train"]) | roots_in(split_rows["test"]))
    print(f"corpus roots: {', '.join(all_roots)}")

    meta_by_key = {}
    for root in all_roots:
        for eng_dir in set(ENGINE_DIR.values()):
            for man in sorted((Path(root) / eng_dir).glob("manifest.shard*.csv")):
                with man.open(encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        if r.get("meta"):
                            meta_by_key[(root, eng_dir, r["idx"])] = r["meta"]
    print(f"meta joined for {len(meta_by_key)} synthesised clips")


    splits, seen_phrase = {}, {}
    for split in ("train", "test"):
        rows = split_rows[split]
        recs, paths = [], set()
        for r in rows:
            root = r.get("corpus_root") or primary
            # The manifest's `engine` is the PLAN's name for the model
            # ("multilingual-expressive"); the output directory is the synthesiser's CLI
            # name ("scicom"). They are not the same string.
            eng_dir = ENGINE_DIR.get(r["engine"], r["engine"])
            src = Path(root) / eng_dir / r["audio_filepath"]
            if not src.exists():
                raise SystemExit(f"missing clip: {src}")
            rel = f"{root_tag(root)}{eng_dir}/{r['audio_filepath']}"
            paths.add(rel)
            recs.append({
                "id": f"{args.config_name}_{root_tag(root)}{r['lang']}_{r['idx']}_{r['engine']}",
                "audio": {"bytes": flac_bytes(src), "path": rel},
                "phrase": r["phrase"], "lang": r["lang"],
                "engine": r["engine"], "voice": r.get("voice") or "",
                "observed_count": int(r.get("count") or 0),
                "duration_s": float(r["duration_s"]),
                "cer": float(r["cer"]) if r.get("cer") else -1.0,
                "asr_hypothesis": r.get("hyp", ""),
                "split": split,
                "meta": r.get("meta") or meta_by_key.get((root, eng_dir, r["idx"]), ""),
            })
            seen_phrase.setdefault(split, set()).add((norm(r["phrase"]), r["lang"]))
        # A counter reset inside a loop once produced 4,694 manifest rows over 2,908 files
        # (CLAUDE.md); assert the row/file/id counts agree rather than trust it.
        assert len(recs) == len(paths) == len({r["id"] for r in recs}), (
            f"{split}: {len(recs)} rows, {len(paths)} paths, {len({r['id'] for r in recs})} ids")
        splits[split] = recs
        print(f"{split}: {len(recs)} clips, {len({r['lang'] for r in recs})} languages")

    missing_meta = sum(1 for recs in splits.values() for r in recs if not r["meta"])
    print(f"VERIFY clips missing meta        : {missing_meta}")
    if missing_meta:
        raise SystemExit("refusing to publish clips without provenance metadata")

    overlap = seen_phrase["train"] & seen_phrase["test"]
    leaked = {p for p, _ in seen_phrase["train"] if p in protected}
    print(f"VERIFY phrase overlap train/test : {len(overlap)}")
    print(f"VERIFY benchmark phrases in train: {len(leaked)}")
    if overlap or leaked:
        raise SystemExit("refusing to build a release with a leaking split")

    d = args.out / "data" / args.config_name
    d.mkdir(parents=True, exist_ok=True)
    # Stale shards from a previous build would be picked up by the Hub's `train-*.parquet`
    # glob alongside the new ones and silently double the split.
    for old_file in d.glob("*.parquet"):
        old_file.unlink()

    # A single 812 MB file with one row group per file is unreadable by the dataset viewer:
    # "Scan size limit exceeded: attempted to read 812011332 bytes, limit is 300000000".
    # Audio bytes live inline, so the fix is both axes -- shard the file AND keep row groups
    # small -- plus a page index so a reader can seek instead of decompressing a whole group.
    for split, recs in splits.items():
        n_files = max(1, -(-len(recs) // args.rows_per_file))
        total = 0
        for i in range(n_files):
            chunk = recs[i * args.rows_per_file:(i + 1) * args.rows_per_file]
            target = d / f"{split}-{i:05d}-of-{n_files:05d}.parquet"
            pq.write_table(table(chunk, feats), target, compression="zstd",
                           use_dictionary=[], row_group_size=args.row_group_size,
                           write_page_index=True)
            total += target.stat().st_size
        biggest = max(f.stat().st_size for f in d.glob(f"{split}-*.parquet"))
        print(f"  {split:<6} {len(recs):>6} rows  {total/1e6:>7.1f} MB in {n_files} file(s), "
              f"largest {biggest/1e6:.1f} MB, row groups of {args.row_group_size}")
        if biggest > 300e6:
            raise SystemExit(f"{split}: a shard is {biggest/1e6:.0f} MB, over the viewer limit")

    stats = {
        "config": args.config_name,
        "splits": {s: {"clips": len(r),
                       "languages": len({x['lang'] for x in r}),
                       "hours": round(sum(x["duration_s"] for x in r) / 3600, 3),
                       "engines": dict(Counter(x["engine"] for x in r)),
                       "voices": len({x["voice"] for x in r if x["voice"]})}
                   for s, r in splits.items()},
        "phrase_overlap_train_test": len(overlap),
        "benchmark_phrases_in_train": len(leaked),
    }
    (args.out / "lexicon_synth_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"\nstaged -> {args.out}")


if __name__ == "__main__":
    main()
