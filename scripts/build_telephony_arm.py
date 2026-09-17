#!/usr/bin/env python3
"""Build a PAIRED narrowband-telephony variant of the existing arms.

Why: every clip gathered so far is ~16 kHz and reasonably clean, while the production
regime is 8 kHz codec-compressed telephony. If hallucination behaviour differs between the
two, every number measured on the wideband arms transfers badly — and nothing published
tells us whether it does.

The fix that survives a replicability constraint: do not go find private call audio, apply a
deterministic transform to the PUBLIC audio we already have. Anyone can reproduce it from
the same inputs with the same script.

Chain: 16 kHz -> band-limit -> 8 kHz -> G.711 mu-law quantise -> back to 16 kHz
(Whisper's required input rate). That is the standard telephone path: narrowband plus
8-bit companding.

Design is PAIRED — same clip ids as the source arm — so wideband vs narrowband is a matched
comparison rather than two different samples, which is both stronger evidence and cheaper.

  python scripts/build_telephony_arm.py --per-arm 300
"""
import argparse, csv, random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
NB = 8000

# Arms worth pairing: both sides of the contrast, plus the phrase-critical positives.
SOURCE_ARMS = ["genuine", "genuine_isolated", "silence", "music", "nonspeech",
               "speech_in_noise", "reduplication"]


def mulaw(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711 mu-law companding: 8-bit quantise in the log domain, then expand back.

    This is the lossy step that actually characterises a phone line -- plain downsampling
    alone would understate the damage.
    """
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * 255)                      # 8 bits
    y = q / 255 * 2 - 1
    return np.sign(y) * (1 / mu) * ((1 + mu) ** np.abs(y) - 1)


def telephony(x: np.ndarray) -> np.ndarray:
    sos = butter(8, [300 / (SR / 2), 3400 / (SR / 2)], btype="band", output="sos")
    x = sosfilt(sos, x)                                   # 300-3400 Hz passband
    x = resample_poly(x, 1, 2)                            # -> 8 kHz
    x = mulaw(np.clip(x, -1, 1))
    x = resample_poly(x, 2, 1)                            # -> 16 kHz for Whisper
    peak = float(np.abs(x).max())
    return np.clip(x / peak * 0.707 if peak > 0 else x, -1, 1).astype("float32")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-arm", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "telephony")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    flac = args.out / "flac"; flac.mkdir(parents=True, exist_ok=True)
    rows = []

    for arm in SOURCE_ARMS:
        man = ROOT / "audio" / arm / "manifest.csv"
        if not man.exists():
            print(f"  (skip {arm})"); continue
        src = list(csv.DictReader(man.open(encoding="utf-8")))
        # Keep every phrase-carrying clip; they are scarce and are the whole point.
        keep = [r for r in src if r.get("has_target_phrase") == "true"]
        rest = [r for r in src if r.get("has_target_phrase") != "true"]
        rng.shuffle(rest)
        sel = keep + rest[: max(0, args.per_arm - len(keep))]
        for r in sel:
            p = ROOT / "audio" / arm / r["audio_filepath"]
            try:
                x, sr = sf.read(p, dtype="float32")
            except Exception:
                continue
            if sr != SR or x.size < SR // 20:
                continue
            y = telephony(x)
            stem = f"{arm}__{Path(r['audio_filepath']).stem}"
            sf.write(flac / f"{stem}.flac", y, SR, format="FLAC", subtype="PCM_16")
            rows.append({
                "audio_filepath": f"flac/{stem}.flac",
                "source_arm": arm,
                "paired_id": Path(r["audio_filepath"]).stem,   # join key back to the wideband clip
                "text": r.get("text") or r.get("reference_text") or "",
                "reference_text": r.get("text") or r.get("reference_text") or "",
                "lang": r.get("lang", ""),
                "duration_s": round(len(y) / SR, 3),
                "has_target_phrase": r.get("has_target_phrase", "false"),
                "target_phrases": r.get("target_phrases", ""),
                "license": r.get("license", ""),
                "note": "G.711 mu-law narrowband simulation of the paired wideband clip",
            })
        print(f"  {arm:<18} {len([x for x in rows if x['source_arm']==arm]):>5} clips "
              f"({len(keep)} phrase-carrying kept)")

    man = args.out / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    h = sum(r["duration_s"] for r in rows) / 3600
    print(f"\ntelephony arm: {len(rows)} clips, {h:.2f} h -> {args.out}")
    print(f"phrase-carrying: {sum(r['has_target_phrase']=='true' for r in rows)}")


if __name__ == "__main__":
    main()
