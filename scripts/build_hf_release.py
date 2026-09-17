#!/usr/bin/env python3
"""Stage the public HF dataset release under a build dir.

Writes one parquet per config plus the vendored upstream lexicons and their licences,
so the release is self-contained and survives an upstream repo disappearing.

Audio is re-encoded wav -> FLAC (lossless) to cut the download roughly in half.

  python scripts/build_hf_release.py --out build/hf
"""
import argparse, csv, io, json, shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Features, Value

ROOT = Path(__file__).resolve().parent.parent
SR = 16000


def flac_bytes(path: Path) -> bytes:
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR}"
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


# datasets>=5 needs torchcodec to ENCODE an Audio column, but the on-disk format is just
# struct<bytes, path> plus a `huggingface` schema-metadata blob. Writing that directly with
# pyarrow produces a byte-identical layout and keeps torchcodec out of the dependency set.
_ARROW = {
    "string": pa.string(), "int32": pa.int32(), "int64": pa.int64(),
    "float32": pa.float32(), "float64": pa.float64(), "bool": pa.bool_(),
}
_AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _table(recs: list[dict], feats: Features) -> pa.Table:
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
        b"huggingface": json.dumps({"info": {"features": feats.to_dict()}}).encode()
    })
    return pa.Table.from_arrays(arrays, schema=schema)


def _coerce(raw, spec):
    if isinstance(spec, Audio):
        return raw
    if spec.dtype.startswith("int"):
        return int(raw) if raw not in ("", None) else 0
    if spec.dtype.startswith("float"):
        return float(raw) if raw not in ("", None) else 0.0
    if spec.dtype == "bool":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return raw or ""


def flac_config(arm: str, extra: dict[str, Value]) -> pa.Table:
    """Like audio_config, but the arm already holds FLAC (harvested, not synthesised),
    so the bytes pass through untouched -- no second lossy-free re-encode."""
    base = ROOT / "audio" / arm
    rows = list(csv.DictReader((base / "manifest.csv").open(encoding="utf-8")))
    feats = Features({"id": Value("string"), "audio": Audio(sampling_rate=SR), **extra})
    recs = []
    for r in rows:
        rel = Path(r["audio_filepath"])
        recs.append({
            "id": rel.stem,
            "audio": {"bytes": (base / rel).read_bytes(), "path": rel.name},
            **{k: _coerce(r.get(k, ""), v) for k, v in extra.items()},
        })
    return _table(recs, feats)


def audio_config(arm: str, extra: dict[str, Value]) -> pa.Table:
    """Build an audio config from an arm's manifest.csv."""
    base = ROOT / "audio" / arm
    rows = list(csv.DictReader((base / "manifest.csv").open(encoding="utf-8")))
    feats = Features({"id": Value("string"), "audio": Audio(sampling_rate=SR), **extra})
    recs = []
    for r in rows:
        rel = Path(r["audio_filepath"])
        recs.append({
            "id": rel.stem,
            "audio": {"bytes": flac_bytes(base / rel), "path": rel.with_suffix(".flac").name},
            **{k: _coerce(r[k], v) for k, v in extra.items()},
        })
    return _table(recs, feats)


def csv_config(path: Path, feats: Features) -> pa.Table:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    recs = [{k: _coerce(r.get(k, ""), v) for k, v in feats.items()} for r in rows]
    return _table(recs, feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "hf")
    args = ap.parse_args()
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)

    configs: dict[str, pa.Table] = {}

    print("[silence] encoding")
    configs["silence"] = audio_config("silence", {
        "floor": Value("string"), "duration_s": Value("float32"), "reference_text": Value("string"),
    })

    print("[reduplication] encoding")
    configs["reduplication"] = audio_config("reduplication", {
        "pattern": Value("string"), "unit": Value("string"), "n_repeats": Value("int32"),
        "rate_hz": Value("float32"), "tail_silence_s": Value("float32"),
        "duration_s": Value("float32"), "reference_text": Value("string"),
        "reference_collapsed": Value("string"),
    })

    GENUINE_FIELDS = {
        "lang": Value("string"), "source": Value("string"), "text": Value("string"),
        "duration_s": Value("float32"), "license": Value("string"),
        "has_target_phrase": Value("bool"), "target_phrases": Value("string"),
        "note": Value("string"), "author": Value("string"), "source_url": Value("string"),
    }
    if (ROOT / "audio" / "genuine" / "manifest.csv").exists():
        print("[genuine] packing")
        configs["genuine"] = flac_config("genuine", GENUINE_FIELDS)
    if (ROOT / "audio" / "genuine_isolated" / "manifest.csv").exists():
        print("[genuine_isolated] packing")
        configs["genuine_isolated"] = flac_config("genuine_isolated", GENUINE_FIELDS)
    if (ROOT / "audio" / "librispeech_test_clean" / "manifest.csv").exists():
        print("[librispeech_test_clean] packing")
        configs["librispeech_test_clean"] = flac_config("librispeech_test_clean", GENUINE_FIELDS)

    if (ROOT / "audio" / "music" / "manifest.csv").exists():
        print("[music] packing")
        configs["music"] = flac_config("music", {
            "title": Value("string"), "artist": Value("string"), "album_title": Value("string"),
            "excerpt_s": Value("float32"), "duration_s": Value("float32"),
            "reference_text": Value("string"), "vocals": Value("string"),
            "license": Value("string"), "source": Value("string"), "source_url": Value("string"),
        })
    if (ROOT / "audio" / "nonspeech" / "manifest.csv").exists():
        print("[nonspeech] packing")
        configs["nonspeech"] = flac_config("nonspeech", {
            "kind": Value("string"), "fsd50k_id": Value("string"),
            "fsd50k_labels": Value("string"), "duration_s": Value("float32"),
            "reference_text": Value("string"), "license": Value("string"),
            "source": Value("string"), "source_url": Value("string"),
        })
    if (ROOT / "audio" / "speech_in_noise" / "manifest.csv").exists():
        print("[speech_in_noise] packing")
        configs["speech_in_noise"] = flac_config("speech_in_noise", {
            "lang": Value("string"), "text": Value("string"), "snr_db": Value("int32"),
            "background": Value("string"), "duration_s": Value("float32"),
            "has_target_phrase": Value("bool"), "target_phrases": Value("string"),
            "speech_source": Value("string"), "license": Value("string"), "note": Value("string"),
        })

    print("[lexicon] loading")
    configs["lexicon"] = csv_config(ROOT / "lexicon" / "combined_lexicon.csv", Features({
        "phrase_norm": Value("string"), "phrase": Value("string"), "lang": Value("string"),
        "count": Value("int64"), "source": Value("string"),
    }))

    if (ROOT / "phrases" / "ban_candidates.csv").exists():
        print("[ban_candidates] loading")
        configs["ban_candidates"] = csv_config(ROOT / "phrases" / "ban_candidates.csv", Features({
            "phrase": Value("string"), "phrase_norm": Value("string"), "lang": Value("string"),
            "hallucination_count": Value("int64"), "genuine_exact": Value("int64"),
            "genuine_substring": Value("string"), "n_words": Value("int32"),
            "verdict": Value("string"), "reason": Value("string"),
        }))

    print("[targets] loading")
    configs["targets"] = csv_config(ROOT / "phrases" / "targets.csv", Features({
        "phrase": Value("string"), "lang": Value("string"), "source": Value("string"),
        "observed_hallucination_count": Value("int64"), "note": Value("string"),
    }))

    print("[malaysian_sources] loading")
    configs["malaysian_sources"] = csv_config(ROOT / "sources" / "malaysian_speech_registry.csv", Features({
        "dataset": Value("string"), "host": Value("string"), "languages": Value("string"),
        "license": Value("string"), "gated": Value("string"), "role": Value("string"),
        "has_ground_truth_text": Value("string"), "notes": Value("string"),
    }))

    # Audio arms are EVALUATION data -> `test`. Declaring them `train` invites the one
    # mistake that invalidates every reported number: fine-tuning on the benchmark.
    # The remaining configs are lookup tables, not splits, and keep `train`.
    AUDIO_ARMS = {"silence", "music", "nonspeech", "reduplication", "speech_in_noise",
                  "genuine", "genuine_isolated", "librispeech_test_clean"}
    for name, tbl in configs.items():
        d = out / "data" / name
        d.mkdir(parents=True, exist_ok=True)
        split = "test" if name in AUDIO_ARMS else "train"
        target = d / f"{split}-00000-of-00001.parquet"
        # FLAC/wav bytes are already compressed and every value is unique, so dictionary
        # encoding only overflows and falls back to PLAIN, storing the payload twice.
        has_audio = "audio" in tbl.column_names
        pq.write_table(tbl, target, compression="zstd",
                       use_dictionary=[] if has_audio else True)
        mb = target.stat().st_size / 1e6
        print(f"  {name:<24} {split:<6} {tbl.num_rows:>6} rows  {mb:>7.1f} MB")

    # Vendored upstream lexicons + the licences that permit redistributing them.
    raw = out / "lexicon_raw"
    (raw / "granary").mkdir(parents=True)
    for f in ["agh_boh.csv", "agh_hallucination_list.csv", "hf_whisper_hallucinations_phrases.csv"]:
        shutil.copy(ROOT / "lexicon" / f, raw / f)
    for f in sorted((ROOT / "lexicon" / "granary").glob("*.txt")):
        shutil.copy(f, raw / "granary" / f.name)
    shutil.copy(ROOT / "lexicon" / "granary" / "DetectWhisperHallucinationFeatures.reference.py",
                raw / "granary" / "DetectWhisperHallucinationFeatures.reference.py")

    # HALAS is CC BY 4.0; redistributing with attribution is permitted and it is the only
    # public set with human LOOPING labels, so keep a copy rather than a dangling pointer.
    ext = out / "external"
    ext.mkdir()
    shutil.copy(ROOT / "manifests" / "halas_dataset.csv", ext / "halas_dataset.csv")
    shutil.copy(ROOT / "manifests" / "halas_README.md", ext / "halas_UPSTREAM_README.md")

    # Benchmark harness + measured scores ship with the dataset, so the numbers in the card
    # are reproducible from the same repo rather than described only in prose.
    bench_src = ROOT / "bench"
    if bench_src.exists():
        bd = out / "bench"; bd.mkdir(parents=True, exist_ok=True)
        for f in ["run_benchmark.py", "score_benchmark.py", "metrics.py", "scores.json"]:
            if (bench_src / f).exists():
                shutil.copy(bench_src / f, bd / f)
        print("bench harness + scores -> bench/")

    lic = out / "licenses"
    lic.mkdir()
    shutil.copy(ROOT / "lexicon" / "LICENSE.agh-mit", lic / "agh-dsp-MIT.txt")

    # The card lives in the repo (DATASET_CARD.md) because this function rmtree's `out`.
    card = ROOT / "DATASET_CARD.md"
    if card.exists():
        shutil.copy(card, out / "README.md")
        print("card -> README.md")
    else:
        print("!! DATASET_CARD.md missing - the release will have no card")

    print(f"\nstaged -> {out}")
    print("total:", f"{sum(p.stat().st_size for p in out.rglob('*') if p.is_file())/1e6:.1f} MB")


if __name__ == "__main__":
    main()
