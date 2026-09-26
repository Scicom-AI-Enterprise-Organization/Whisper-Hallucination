#!/usr/bin/env python3
"""Pull the training curves out of W&B so they can be plotted next to the results.

The trainer logs every 25 steps to the `whisper-hallucination` project, one run per sweep
configuration, grouped by mix. This dumps {run name: [[step, loss], ...]} so the figure script
does not need network access or a W&B login.

    .venv_wild/bin/python bench/fetch_wandb_loss.py --out bench/wandb_loss.json
"""
import argparse, json, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "aies-scicom-scicom-ai"))
ap.add_argument("--project", default="whisper-hallucination")
ap.add_argument("--out", type=Path, default=Path("bench/wandb_loss.json"))
a = ap.parse_args()

import wandb
api = wandb.Api()
out = {}
for run in api.runs(f"{a.entity}/{a.project}"):
    if run.state not in ("finished", "running"):
        continue
    pts = []
    for row in run.scan_history(keys=["train/global_step", "train/loss"]):
        s, l = row.get("train/global_step"), row.get("train/loss")
        if s is not None and l is not None:
            pts.append([int(s), float(l)])
    if len(pts) < 5:
        continue
    pts.sort()
    out[run.name] = {"points": pts, "group": run.group, "config": {
        k: run.config.get(k) for k in ("mix", "method", "lora_r", "lr", "blank_share")}}
    print(f"{run.name:<32} {len(pts):>4} points  final {pts[-1][1]:.3f}", flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out))
print(f"-> {a.out}  ({len(out)} runs)")
