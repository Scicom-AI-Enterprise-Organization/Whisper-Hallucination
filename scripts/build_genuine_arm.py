#!/usr/bin/env python3
"""Build the genuine-speech (positive) arm from permissively-licensed public corpora.

This is the counterpart to `silence` and `reduplication`. Those measure what the model
INVENTS; this measures what a mitigation DESTROYS. Without it, "hallucination rate fell"
is unfalsifiable -- a decoder that outputs nothing scores perfectly on the negative arms.

Sources are restricted to licences that permit redistribution (CC0 / CC BY). Each row
keeps its own `source` and `license`, and `has_target_phrase` marks the clips that
genuinely contain one of the high-risk phrases from phrases/targets.csv -- those are the
true minimal pairs against the hallucinated negatives.

  python scripts/build_genuine_arm.py --sources sarawak manglish fleurs_ms
"""
import argparse, csv, io, os, re, sys, unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
SR = 16000

SOURCES = {
    # key: (repo_id, [parquet paths], text_col, lang, license, note)
    "sarawak": ("SaLTUNIMAS/sarawak-malay-asr",
                ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],
                "transcription", "ms", "CC BY 4.0", "Sarawak Malay, human transcripts"),
    "manglish": ("emhaihsan/Synth-Manglish",
                 ["parquet/train-00000-of-00002.parquet", "parquet/train-00001-of-00002.parquet"],
                 "text", "ms", "CC BY 4.0", "synthetic Manglish code-switching (TTS voice)"),
    # English WER regression guard. Same split the hallucination-mitigation literature
    # reports on (arXiv:2609.04561 baseline large-v3 = 4.06% WER), so our cost numbers
    # are directly comparable to theirs rather than only to our own Malaysian arms.
    "librispeech": ("openslr/librispeech_asr", ["clean/test/0000.parquet"],
                    "text", "en", "CC BY 4.0", "LibriSpeech test-clean, read English audiobooks"),
    "fleurs_ms": ("google/fleurs",
                  ["parquet-data/ms_my/test-00000-of-00001.parquet",
                   "parquet-data/ms_my/validation-00000-of-00001.parquet"],
                  "raw_transcription", "ms", "CC BY 4.0", "human-read Wikipedia prose"),
}

_PUNCT = re.compile(r"[.?!,;:\"'`´()\[\]…]+")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "").strip().casefold()
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", t)).strip()


def load_targets() -> dict[str, list[str]]:
    by_lang: dict[str, list[str]] = {}
    for r in csv.DictReader((ROOT / "phrases" / "targets.csv").open(encoding="utf-8")):
        by_lang.setdefault(r["lang"], []).append(norm(r["phrase"]))
    return by_lang


def to_flac(raw: bytes, dest: Path) -> float | None:
    try:
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        return None
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    if x.size < SR // 20:          # under 50 ms is not a usable utterance
        return None
    peak = float(np.abs(x).max())
    if peak <= 0:
        return None
    x = np.clip(x / peak * 0.707, -1.0, 1.0).astype("float32")
    sf.write(dest, x, SR, format="FLAC", subtype="PCM_16")
    return len(x) / SR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "genuine")
    ap.add_argument("--limit-per-source", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    targets = load_targets()
    flac_dir = args.out / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)
    man = args.out / "manifest.csv"
    rows = list(csv.DictReader(man.open(encoding="utf-8"))) if man.exists() else []
    have = {r["audio_filepath"] for r in rows}
    print(f"[resume] {len(rows)} existing rows")

    for key in args.sources:
        repo, paths, tcol, lang, lic, note = SOURCES[key]
        tgt = targets.get(lang, [])
        print(f"\n=== {key}  ({repo})")
        made = hits = skipped = 0
        n = 0   # per-SOURCE, not per-shard: resetting here made shard 2 overwrite shard 1's files
        for path in paths:
            print(f"  downloading {path}")
            local = hf_hub_download(repo, path, repo_type="dataset", token=tok)
            f = pq.ParquetFile(local)
            cols = [x.name for x in f.schema_arrow]
            acol = "audio" if "audio" in cols else None
            if not acol or tcol not in cols:
                print(f"    !! missing audio/{tcol} in {cols}"); continue
            for batch in f.iter_batches(batch_size=64, columns=[acol, tcol]):
                for rec in batch.to_pylist():
                    if args.limit_per_source and made >= args.limit_per_source:
                        break
                    au, text = rec[acol], (rec[tcol] or "")
                    if not isinstance(au, dict) or not au.get("bytes") or not text.strip():
                        skipped += 1; continue
                    n += 1
                    stem = f"{key}_{n:06d}"
                    name = f"flac/{stem}.flac"
                    if name in have:
                        continue
                    dur = to_flac(au["bytes"], flac_dir / f"{stem}.flac")
                    if dur is None:
                        skipped += 1; continue
                    nt = norm(text)
                    matched = [p for p in tgt if p and p in nt]
                    hits += bool(matched)
                    made += 1
                    rows.append({
                        "audio_filepath": name, "lang": lang, "source": key,
                        "text": text.strip()[:2000], "duration_s": round(dur, 3),
                        "license": lic, "has_target_phrase": str(bool(matched)).lower(),
                        "target_phrases": "|".join(sorted(set(matched))), "note": note,
                        "author": repo, "source_url": f"https://huggingface.co/datasets/{repo}",
                    })
                    if made % 400 == 0:
                        print(f"    {made} clips, {hits} with a target phrase", flush=True)
                if args.limit_per_source and made >= args.limit_per_source:
                    break
        print(f"  {key}: {made} clips, {hits} with a target phrase, {skipped} skipped")

        fields = ["audio_filepath", "lang", "source", "text", "duration_s", "license",
                  "has_target_phrase", "target_phrases", "note", "author", "source_url"]
        with man.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    import collections
    print(f"\ntotal {len(rows)} clips, {sum(float(r['duration_s']) for r in rows)/3600:.2f} h")
    print("by source:", dict(collections.Counter(r["source"] for r in rows)))
    print("with target phrase:", sum(r["has_target_phrase"] == "true" for r in rows))
    print(f"manifest -> {man}")


if __name__ == "__main__":
    main()
