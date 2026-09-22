#!/usr/bin/env python3
"""Join the HALAS human annotations to their Earnings-22 audio and write a wild arm.

HALAS (`manifests/halas_dataset.csv`, 3,611 rows, CC BY 4.0) is the only set here where a
human looked at real recordings and marked which ASR outputs were hallucinated, span by span,
across nine systems. The audio is Earnings-22, fetched separately by
`scripts/fetch_audio.py halas`.

The join key is not stored anywhere: HALAS names a clip `{segment_id}_{file_id}.wav`, which
is exactly the pair the distil-whisper/earnings22 `chunked` parquets carry as two columns.

Each output row keeps the human verdict per model, so a checkpoint can be scored against what
annotators actually saw rather than against a heuristic:

  reason=halas_hallucination   at least one of the nine systems was marked
                               "Hallucination or looping" on this clip
  reason=halas_clean           no system was

    python scripts/build_halas_arm.py --out audio_wild/halas/earnings22
"""
import argparse, csv, glob, io, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
# Exactly as the CSV spells them. Guessing these (canary_1b, parakeet_tdt_v2, phi_4,
# crisperwhisper) silently yields 0 flagged for those systems instead of an error.
MODEL_COLS = ["whisper_large_v2", "whisper_large_v3", "whisper_large_v3_turbo",
              "crisper_whisper", "canary", "canary_flash", "parakeet", "phi4", "granite"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "manifests" / "halas_dataset.csv")
    ap.add_argument("--audio", type=Path, default=Path("audio/halas_earnings22"))
    ap.add_argument("--out", type=Path, default=Path("audio_wild/halas/earnings22"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    labels = {}
    with args.labels.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [m for m in MODEL_COLS if f"{m}_label" not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"label columns not in {args.labels.name}: {missing}\n"
                     f"available: {[c for c in reader.fieldnames if c.endswith('_label')]}")
        for r in reader:
            labels[r["audio_id"]] = r
    print(f"[halas] {len(labels)} annotated clips", flush=True)

    files = sorted(glob.glob(str(args.audio / "chunked" / "**" / "*.parquet"), recursive=True))
    print(f"[halas] scanning {len(files)} parquet shards", flush=True)

    (args.out / "wav").mkdir(parents=True, exist_ok=True)
    rows, matched, seen = [], 0, 0
    for path in files:
        tbl = pq.read_table(path, columns=["file_id", "segment_id", "audio", "transcription"])
        d = tbl.to_pydict()
        for fid, sid, aud, tr in zip(d["file_id"], d["segment_id"], d["audio"], d["transcription"]):
            seen += 1
            key = f"{sid}_{fid}.wav"
            lab = labels.get(key)
            if lab is None:
                continue
            try:
                x, sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32")
            except Exception:
                continue
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=SR)
            flagged = [m for m in MODEL_COLS
                       if str(lab.get(f"{m}_label", "")).startswith("Hallucination")]
            rel = f"wav/{key.replace('.wav', '')}.flac"
            sf.write(args.out / rel, x, SR, format="FLAC", subtype="PCM_16")
            rows.append({
                "id": key.replace(".wav", ""), "audio_filepath": rel,
                "source": "distil-whisper/earnings22 + HALAS labels",
                "reasons": "halas_hallucination" if flagged else "halas_clean",
                "n_models_flagged": len(flagged), "models_flagged": "|".join(flagged),
                "reference_text": lab.get("corrected_reference_text") or lab.get("e22_reference_text", ""),
                "duration_s": round(len(x) / SR, 3),
                "hyp": lab.get("whisper_large_v3_prediction", "")[:300],
                "max_token_run": "", "speech_s": "", "speech_frac": "", "rms_db": "",
                "config": "chunked", "source_id": key,
            })
            matched += 1
            if args.limit and matched >= args.limit:
                break
        print(f"  {Path(path).name}: {matched} matched of {seen} scanned", flush=True)
        if args.limit and matched >= args.limit:
            break

    man = args.out / "manifest.csv"
    if rows:
        with man.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    report = {"annotated": len(labels), "scanned": seen, "matched": matched,
              "by_reason": dict(Counter(r["reasons"] for r in rows)),
              "flagged_per_model": {m: sum(m in r["models_flagged"] for r in rows)
                                    for m in MODEL_COLS}}
    (args.out / "halas_report.json").write_text(json.dumps(report, indent=2))
    print(f"[halas] matched {matched}/{len(labels)} -> {man}")
    print(json.dumps(report["by_reason"], indent=1))


if __name__ == "__main__":
    main()
