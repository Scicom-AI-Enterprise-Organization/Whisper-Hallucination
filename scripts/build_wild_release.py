#!/usr/bin/env python3
"""Package the wild-audio arm: real recordings that made an ASR model hallucinate or loop.

Every other audio config in this dataset is a built stimulus -- synthesised silence, FMA
excerpts, generated reduplication. This one is not constructed at all. Two kinds of row:

  halas_*        Earnings-22 clips carrying HALAS human span annotations for nine ASR
                 systems, so `models_flagged` says who hallucinated and `n_models_flagged`
                 how many. Ground truth is a person's judgement.
  loop /         clips mined from streamed corpora by `scripts/mine_wild_hallucinations.py`,
  blank_speech   where the label needs no annotator: a token run >= 6, or words emitted
                 where Silero VAD finds no speech at all.

Ships `train` and `test`. The split is by SOURCE RECORDING, not by clip
(`scripts/split_wild.py`): several AudioSet clips come from one video and a dozen HALAS
segments from one call, so a clip-level split would put near neighbours on both sides. HALAS
is entirely `test` -- it is the only human-labelled part and spending it on training would
cost the benchmark -- and so is every `loop` clip, which has no reference to train toward. The Koenecke aphasia clips are deliberately
NOT here: that is clinical speech from a membership-gated corpus, and consent, not licence,
is the reason. `scripts/fetch_audio.py aphasia` still gets them for local measurement.

Parquet is written with pyarrow directly (torchcodec is not needed to WRITE an audio column),
sharded and page-indexed so the Hub viewer can scan it -- a single large file returns
"Scan size limit exceeded".

    .venv_wild/bin/python scripts/build_wild_release.py --out build/wild
"""
import argparse, csv, json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Features, Value

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
_AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
_ARROW = {"string": pa.string(), "int32": pa.int32(), "int64": pa.int64(),
          "float32": pa.float32(), "float64": pa.float64(), "bool": pa.bool_()}

# Collection -> (upstream dataset, licence as the upstream states it).
PROVENANCE = {
    "ami":            ("edinburghcstr/ami", "CC BY 4.0"),
    "voxpopuli":      ("facebook/voxpopuli", "CC0-1.0"),
    "peoples_speech": ("MLCommons/peoples_speech", "CC BY / CC BY-SA (per clip upstream)"),
    "earnings22":     ("distil-whisper/earnings22", "unspecified upstream"),
    "halas":          ("distil-whisper/earnings22 audio + HALAS labels (CC BY 4.0)",
                       "unspecified upstream audio; labels CC BY 4.0"),
}

FEATURES = Features({
    "id": Value("string"),
    "audio": Audio(sampling_rate=SR),
    "collection": Value("string"),
    "source_dataset": Value("string"),
    "license": Value("string"),
    "reasons": Value("string"),
    "reference_text": Value("string"),
    "mined_transcript": Value("string"),
    "n_models_flagged": Value("int32"),
    "models_flagged": Value("string"),
    "max_token_run": Value("int32"),
    "speech_frac": Value("float32"),
    "rms_db": Value("float32"),
    "duration_s": Value("float32"),
    "split": Value("string"),
})


def table(recs) -> pa.Table:
    fields, arrays = [], []
    for name, spec in FEATURES.items():
        if isinstance(spec, Audio):
            fields.append(pa.field(name, _AUDIO_TYPE))
            arrays.append(pa.array([r[name] for r in recs], type=_AUDIO_TYPE))
        else:
            t = _ARROW[spec.dtype]
            fields.append(pa.field(name, t))
            arrays.append(pa.array([r[name] for r in recs], type=t))
    schema = pa.schema(fields, metadata={
        b"huggingface": json.dumps({"info": {"features": FEATURES.to_dict()}}).encode()})
    return pa.Table.from_arrays(arrays, schema=schema)


def num(v, cast=float, default=0.0):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wild", type=Path, default=Path("audio_wild"))
    ap.add_argument("--out", type=Path, default=Path("build/wild"))
    ap.add_argument("--config-name", default="wild")
    ap.add_argument("--exclude", nargs="*", default=["aphasia", "gigaspeech"],
                    help="collections left out. `aphasia` is clinical speech and is never "
                         "re-hosted. `gigaspeech` (config l) is a duplicate of `gigaspeech_xs`: "
                         "streaming either config from the start yields the same first 8,000 "
                         "clips, and the two runs matched 33/33 on source_id.")
    # AudioSet clips are 10 s each where HALAS clips are 1-3 s, so a row here is ~196 KB and
    # 3,000 rows overshot the viewer's 300 MB scan limit at 588 MB. Size the shard for the
    # LONGEST collection, not the average.
    ap.add_argument("--rows-per-file", type=int, default=600)
    ap.add_argument("--row-group-size", type=int, default=150)
    args = ap.parse_args()

    recs, skipped = [], Counter()
    for man in sorted(args.wild.rglob("manifest.csv")):
        collection = man.parent.parent.name
        if collection in args.exclude:
            skipped[collection] += 1
            continue
        source, lic = PROVENANCE.get(collection, ("unknown", "unspecified"))
        with man.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                path = man.parent / r["audio_filepath"]
                if not path.exists():
                    continue
                recs.append({
                    "id": f"{collection}_{r.get('id') or path.stem}",
                    "audio": {"bytes": path.read_bytes(), "path": f"{collection}/{path.name}"},
                    "collection": collection,
                    "source_dataset": source,
                    "license": lic,
                    "reasons": r.get("reasons", ""),
                    "reference_text": r.get("reference_text", ""),
                    "mined_transcript": r.get("hyp", ""),
                    "n_models_flagged": int(num(r.get("n_models_flagged"), float, 0)),
                    "models_flagged": r.get("models_flagged", ""),
                    "max_token_run": int(num(r.get("max_token_run"), float, 0)),
                    "speech_frac": num(r.get("speech_frac"), float, -1.0),
                    "rms_db": num(r.get("rms_db"), float, 0.0),
                    "duration_s": num(r.get("duration_s"), float, 0.0),
                    "split": (r.get("split") or "test"),
                })
    if not recs:
        raise SystemExit("no wild clips found")

    ids = {r["id"] for r in recs}
    assert len(ids) == len(recs), f"{len(recs)} rows but {len(ids)} unique ids"

    d = args.out / "data" / args.config_name
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.parquet"):
        old.unlink()
    for split in ("train", "test"):
        part = [r for r in recs if r["split"] == split]
        if not part:
            continue
        n_files = max(1, -(-len(part) // args.rows_per_file))
        for i in range(n_files):
            chunk = part[i * args.rows_per_file:(i + 1) * args.rows_per_file]
            target = d / f"{split}-{i:05d}-of-{n_files:05d}.parquet"
            pq.write_table(table(chunk), target, compression="zstd", use_dictionary=[],
                           row_group_size=args.row_group_size, write_page_index=True)
    n_files = len(list(d.glob("*.parquet")))
    biggest = max(f.stat().st_size for f in d.glob("*.parquet"))
    if biggest > 300e6:
        raise SystemExit(f"a shard is {biggest/1e6:.0f} MB, over the viewer's scan limit")

    stats = {
        "config": args.config_name, "clips": len(recs),
        "hours": round(sum(r["duration_s"] for r in recs) / 3600, 3),
        "splits": dict(Counter(r["split"] for r in recs)),
        "by_collection": dict(Counter(r["collection"] for r in recs)),
        "by_reason": dict(Counter(x for r in recs for x in (r["reasons"] or "").split("|") if x)),
        "excluded_collections": dict(skipped),
        "files": n_files, "largest_file_mb": round(biggest / 1e6, 1),
    }
    (args.out / "wild_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"staged -> {args.out}")


if __name__ == "__main__":
    main()
