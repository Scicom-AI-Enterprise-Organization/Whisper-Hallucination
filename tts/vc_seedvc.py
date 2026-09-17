#!/usr/bin/env python3
"""seed-vc V1: diffusion-transformer zero-shot conversion from a single reference clip.

Driven through the repo's own SeedVCWrapper rather than `inference.py`, because the CLI
reloads every model per call and this grid is 200+ conversions per target.

`convert_voice` is a generator that streams chunks and returns the full waveform last; the
non-stream path still yields once, so take the final item.

    .venv_seedvc/bin/python tts/vc_seedvc.py --device cuda:6
"""
import argparse, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vc_common import ROOT, Writer, load_sources, load_targets  # noqa: E402


def last_wave(gen):
    """Drain the wrapper's generator and take the full clip.

    With `stream_output=False` it never yields -- it RETURNS the waveform, which lands in
    StopIteration.value, not in the iteration. Draining with a plain for-loop silently gets
    nothing, which is what "0 ok, 8 failed" looked like on the first run.
    """
    out = None
    try:
        while True:
            out = next(gen)
    except StopIteration as stop:
        if stop.value is not None:
            out = stop.value
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_out")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--diffusion-steps", type=int, default=25)
    ap.add_argument("--repo", type=Path, default=ROOT / "vc_repos" / "seed-vc")
    ap.add_argument("--limit", type=int, default=0, help="first N source clips only (smoke test)")
    ap.add_argument("--ref", choices=["short", "long"], default="short",
                    help="one-shot reference: 6 s (the realistic case) or the full ~20 s clip")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(args.repo))
    import os
    os.chdir(args.repo)                       # the repo resolves configs relative to cwd
    from seed_vc_wrapper import SeedVCWrapper
    vc = SeedVCWrapper(device=torch.device(args.device))

    sources, targets = load_sources(args.sources), load_targets(args.targets)
    if args.limit:
        sources = sources[:args.limit]
    refkey = "ref_path" if args.ref == "long" else "ref_short_path"
    w = Writer(args.out, "seedvc", args.sources.name)
    print(f"[seedvc] {len(sources)} sources x {len(targets)} targets", flush=True)

    for t in targets:
        for i, s in enumerate(sources):
            try:
                res = last_wave(vc.convert_voice(
                    source=s["abs_path"], target=t[refkey],
                    diffusion_steps=args.diffusion_steps, length_adjust=1.0,
                    inference_cfg_rate=0.7, f0_condition=False, auto_f0_adjust=False,
                    pitch_shift=0, stream_output=False))
                # f0_condition=False -> the wrapper vocodes at 22.05 kHz.
                sr, wave = res if isinstance(res, tuple) else (22050, res)
                if wave is None:
                    raise RuntimeError("wrapper returned no audio")
                w.add(s, t, wave, sr, i)
            except Exception as e:
                traceback.print_exc()
                w.add(s, t, None, 22050, i, error=f"{type(e).__name__}: {e}")
    w.close()


if __name__ == "__main__":
    main()
