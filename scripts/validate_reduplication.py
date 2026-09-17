#!/usr/bin/env python3
"""QA gate for the reduplication arm: the audio must actually contain the number of
repeats the manifest claims, otherwise "Whisper over-generated N repeats" is unfalsifiable.

The clip layout is deterministic (0.25 s lead, then n units at a known period, then the
tail), so this checks energy against the expected grid rather than running blind onset
detection -- amplitude thresholding splits a CV syllable in two, because the plosive
release is a separate transient from the voiced nucleus.

Checks per clip:
  1. each of the n expected unit windows carries energy
  2. the gap between consecutive units is quieter than the units themselves
  3. nothing sounds after the last unit (a spurious n+1th unit would invalidate the label)
"""
import csv, sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
LEAD_S = 0.25      # must match make_reduplication_arm.build_clip
DUTY = 0.62


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def check(path: Path, n: int, rate: float, tail_s: float) -> str | None:
    x, sr = sf.read(path)
    period, dur = 1.0 / rate, DUTY / rate
    # Mirror build_clip's integer arithmetic exactly. Recomputing edges from float seconds
    # drifts ~1 sample per unit against the generator's cumulative int() blocks, which is
    # enough to push a short click across a window boundary.
    lead = int(LEAD_S * sr)
    unit_len = int(dur * sr)
    block = unit_len + int((period - dur) * sr)
    unit_rms, gap_rms = [], []

    for k in range(n):
        start = lead + k * block
        unit_rms.append(rms(x[start: start + unit_len]))
        g = x[start + unit_len: start + block]
        if g.size:
            gap_rms.append(rms(g))

    if min(unit_rms) <= 0:
        return f"unit {int(np.argmin(unit_rms))} is silent"

    floor = rms(x[: int(LEAD_S * 0.8 * sr)])          # dither-only lead
    if min(unit_rms) < max(floor * 20, 1e-3):
        return f"unit {int(np.argmin(unit_rms))} too quiet ({min(unit_rms):.5f} vs floor {floor:.6f})"
    if gap_rms and max(gap_rms) > min(unit_rms) * 0.5:
        return f"inter-unit gap not distinct (gap {max(gap_rms):.4f} vs unit {min(unit_rms):.4f})"

    # Anything after the final unit must be tail silence only.
    after = x[lead + n * block:]
    if after.size and rms(after) > min(unit_rms) * 0.15:
        return f"energy after the last unit (rms {rms(after):.4f}) - extra repeat?"
    return None


def main() -> int:
    arm = ROOT / "audio" / "reduplication"
    rows = list(csv.DictReader((arm / "manifest.csv").open(encoding="utf-8")))
    bad = []
    for r in rows:
        err = check(arm / r["audio_filepath"], int(r["n_repeats"]), float(r["rate_hz"]), float(r["tail_silence_s"]))
        if err:
            bad.append((r["audio_filepath"], err))
    print(f"{len(rows) - len(bad)}/{len(rows)} clips match their manifest layout")
    for name, err in bad[:15]:
        print(f"  {name}: {err}")
    if len(bad) > 15:
        print(f"  ... and {len(bad) - 15} more")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
