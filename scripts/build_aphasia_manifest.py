#!/usr/bin/env python3
"""Manifest the Koenecke et al. aphasia clips so they can be baselined.

187 de-identified AphasiaBank segments that hallucinated in the FAccT'24 study -- real audio,
confirmed triggers, disordered speech. They sit under `audio/`, not `audio_wild/`, and that is
deliberate: `scripts/build_wild_release.py` only reads `audio_wild/`, so these can never be
swept into a published release by accident. They are clinical speech from a membership-gated
corpus, and consent rather than licence is the reason they stay local.

    python scripts/build_aphasia_manifest.py
"""
import argparse, csv, io
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("audio/aphasia_koenecke"))
    args = ap.parse_args()

    wavs = sorted(p for p in (args.dir / "wav").rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3"))
    if not wavs:
        raise SystemExit(f"no audio under {args.dir}/wav -- run scripts/fetch_audio.py aphasia")

    rows = []
    for p in wavs:
        try:
            info = sf.info(str(p))
            dur = round(info.frames / info.samplerate, 3)
        except Exception:
            dur = 0.0
        rows.append({"id": p.stem, "audio_filepath": str(p.relative_to(args.dir)),
                     "source": "koenecke/hallucination_harms (AphasiaBank, de-identified)",
                     "reasons": "aphasia", "duration_s": dur, "hyp": "",
                     "reference_text": "", "n_models_flagged": "", "models_flagged": "",
                     "max_token_run": "", "speech_s": "", "speech_frac": "", "rms_db": "",
                     "config": "koenecke", "source_id": p.name})
    man = args.dir / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[aphasia] {len(rows)} clips, {sum(r['duration_s'] for r in rows)/60:.1f} min -> {man}")


if __name__ == "__main__":
    main()
