#!/usr/bin/env python3
"""Rebuild a synth manifest from the wavs on disk.

The synthesisers name files deterministically as {lang}_{index:04d}.flac, where index is the
row number in phrases.jsonl, so a lost manifest is recoverable without re-synthesising.
(It was lost to an `rsync --delete` before tts/out was added to the sync excludes.)
"""
import argparse, csv, json
from pathlib import Path

import soundfile as sf

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--phrases", type=Path, default=Path("tts/phrases.jsonl"))
ap.add_argument("--speakers", nargs="+",
                default=["multilingual-tts_audio_Grace", "multilingual-tts_audio_Rahman",
                         "DisfluencySpeech"])
a = ap.parse_args()

items = [json.loads(l) for l in a.phrases.open(encoding="utf-8")]
rows = []
for i, it in enumerate(items):
    stem = f"{it['lang']}_{i:04d}"
    p = a.dir / "wav" / f"{stem}.flac"
    if p.exists():
        info = sf.info(p)
        rows.append({**it, "speaker": a.speakers[i % len(a.speakers)],
                     "audio_filepath": f"wav/{stem}.flac",
                     "duration_s": round(info.frames / info.samplerate, 3), "status": "ok"})
    else:
        rows.append({**it, "speaker": a.speakers[i % len(a.speakers)],
                     "audio_filepath": "", "duration_s": 0.0, "status": "missing"})

man = a.dir / "manifest.csv"
with man.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{sum(r['status']=='ok' for r in rows)}/{len(rows)} recovered -> {man}")
