#!/usr/bin/env python3
"""Calibrate the speaker-similarity scale before anyone reads a ranking off it.

`score_vc.py` reports cosine to the target and to the source, and calls a system "converted"
when the first exceeds the second. That test is worthless without knowing the scale: if two
unrelated speakers already score 0.70 on 1.6 s clips, then 0.758 vs 0.740 is noise, not a
conversion. This measures the two reference points:

  floor    cosine between clips of DIFFERENT speakers (source clips vs each target)
  ceiling  cosine between two halves of the SAME speaker's reference

A conversion is only real if sim_tgt sits near the ceiling and clearly above the floor.

    .venv_bench/bin/python tts/calibrate_vc_sim.py --device cuda:6
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
SV_MODEL = "microsoft/wavlm-base-plus-sv"
SR = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--match-seconds", type=float, default=1.6,
                    help="clip length to measure at; match the converted clips")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_sim_calibration.json")
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, WavLMForXVector
    feat = AutoFeatureExtractor.from_pretrained(SV_MODEL)
    sv = WavLMForXVector.from_pretrained(SV_MODEL).to(args.device).eval()

    def emb(x):
        if isinstance(x, (str, Path)):
            x, sr = sf.read(str(x), dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
        inp = feat([x], sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = sv(**{k: v.to(args.device) for k, v in inp.items()}).embeddings
        return torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()

    targets = json.loads((args.targets / "targets.json").read_text())
    refs = {t["id"]: emb(args.targets / t["id"] / "ref.flac") for t in targets}

    # Ceiling and floor must be DURATION-MATCHED to the converted clips (~1.6 s): x-vector
    # cosine rises with clip length, so a ceiling measured on 9 s halves would make every
    # system look hopeless for a reason that has nothing to do with conversion.
    dur = int(args.match_seconds * SR)
    rng = np.random.default_rng(0)

    def chunks(tid, k=12):
        """Duration-matched chunks drawn from one speaker's pool."""
        out = []
        for f in sorted((args.targets / tid / "pool").glob("*.flac")):
            x, _ = sf.read(f, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            if len(x) >= dur:
                i = rng.integers(0, len(x) - dur + 1)
                out.append(x[i:i + dur])
            if len(out) >= k:
                break
        return out

    per_target = {t["id"]: [emb(c) for c in chunks(t["id"])] for t in targets}

    ceil = []
    for tid, es in per_target.items():
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                ceil.append(float(es[i] @ es[j]))

    cross_chunk = []
    ids = list(per_target)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            for ea in per_target[a][:6]:
                for eb in per_target[b][:6]:
                    cross_chunk.append(float(ea @ eb))

    # Floor: unconverted source clips against each target, and targets against each other.
    rows = [r for r in csv.DictReader((args.sources / "manifest.csv").open(encoding="utf-8"))
            if r.get("status") == "ok"]
    src_e = [emb(args.sources / r["audio_filepath"]) for r in rows]
    floor = [float(e @ refs[t["id"]]) for e in src_e for t in targets]
    cross = [float(refs[a["id"]] @ refs[b["id"]])
             for i, a in enumerate(targets) for b in targets[i + 1:]]

    out = {
        "match_seconds": args.match_seconds,
        "ceiling_same_speaker": {"mean": round(float(np.mean(ceil)), 4),
                                 "p05": round(float(np.percentile(ceil, 5)), 4),
                                 "min": round(float(np.min(ceil)), 4), "n": len(ceil)},
        "floor_diff_speaker_matched": {"mean": round(float(np.mean(cross_chunk)), 4),
                                       "p95": round(float(np.percentile(cross_chunk, 95)), 4),
                                       "n": len(cross_chunk)},
        "floor_source_vs_target": {"mean": round(float(np.mean(floor)), 4),
                                   "p95": round(float(np.percentile(floor, 95)), 4),
                                   "max": round(float(np.max(floor)), 4), "n": len(floor)},
        "floor_target_vs_target": {"mean": round(float(np.mean(cross)), 4),
                                   "max": round(float(np.max(cross)), 4), "n": len(cross)},
    }
    args.out.write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"{k:<26} {v}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
