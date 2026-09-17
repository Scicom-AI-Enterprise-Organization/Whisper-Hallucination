#!/usr/bin/env python3
"""OmniVoice in the cloning arm: render each phrase in each target speaker's voice.

OmniVoice is the *source* for every conversion candidate (`synth_omnivoice.py`, auto mode:
no reference audio), which is why it is the ΔCER baseline. But it also takes a reference
clip -- `generate(text=, ref_audio=, ref_text=)` -- so it can aim at a given speaker just
like Higgs v3 does, and being the generator the 100-language sweep actually runs on, the
number worth having is whether conditioning on a voice costs it intelligibility.

Same shape as `clone_higgs3.py`: text in, target reference in, audio out. It does NOT
convert the source waveform -- the source clip is only the baseline the scorer measures
degradation against, so `← source` is similarity to its own auto-mode rendering.

Language ids are aliased exactly as in `synth_omnivoice.py`: OmniVoice ids are mostly
ISO-639-3 while the lexicon uses Whisper's 2-letter codes, and an unmapped code does not
error -- it silently drops to language-agnostic mode, a quiet quality loss.

    .venv_omni/bin/python tts/clone_omnivoice.py --device cuda:7
"""
import argparse, json, sys, traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vc_common import ROOT, SR, Writer, load_sources, load_targets  # noqa: E402


def as_wave(out):
    """OmniVoice returns a batch; unwrap to 1-D float32 whatever the nesting."""
    x = out[0] if isinstance(out, (list, tuple)) else out
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.asarray(x, dtype="float32").squeeze()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_out")
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--num-step", type=int, default=32, help="diffusion steps")
    ap.add_argument("--alias", type=Path, default=ROOT / "tts" / "omnivoice_lang_alias.json")
    ap.add_argument("--ref", choices=["short", "full"], default="short",
                    help="which target clip to condition on (the other arms use the short one)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    alias = json.loads(args.alias.read_text()) if args.alias.exists() else {}
    if not alias:
        print(f"WARNING: no {args.alias.name}; every phrase falls back to "
              f"language-agnostic mode (see tts/omnivoice_langmap.py)", flush=True)

    torch.manual_seed(args.seed)
    model = OmniVoice.from_pretrained(args.model)
    model = model.to(args.device).eval() if hasattr(model, "to") else model
    gcfg = OmniVoiceGenerationConfig(num_step=args.num_step)

    sr_model = None
    for obj, attr in ((model, "sample_rate"), (getattr(model, "config", None), "sample_rate"),
                      (getattr(model, "config", None), "sampling_rate")):
        if obj is not None and getattr(obj, attr, None):
            sr_model = int(getattr(obj, attr)); break
    sr_model = sr_model or 24000

    sources, targets = load_sources(args.sources), load_targets(args.targets)
    if args.limit:
        sources = sources[:args.limit]
    w = Writer(args.out, "omnivoice_clone", args.sources.name, resume=args.resume)
    print(f"[omnivoice_clone] {len(sources)} phrases x {len(targets)} targets @ {sr_model} Hz",
          flush=True)

    for t in targets:
        ref_path = t["ref_short_path" if args.ref == "short" else "ref_path"]
        ref_text = (t.get("ref_short_text") or "").strip()
        if ref_text.isupper():                 # LibriSpeech transcripts are all-caps
            ref_text = ref_text.lower()

        for i, s in enumerate(sources):
            if w.skip(s, t):
                continue
            try:
                with torch.no_grad():
                    out = model.generate(
                        text=[s["phrase"]],
                        language=[alias.get(s["lang"], s["lang"])],
                        ref_audio=[ref_path],
                        ref_text=[ref_text] if ref_text else None,
                        generation_config=gcfg,
                    )
                x = as_wave(out)
                if x.size < SR // 20:
                    w.add(s, t, None, sr_model, i, error="output shorter than 50 ms")
                    continue
                w.add(s, t, x, sr_model, i)
            except Exception as e:
                traceback.print_exc()
                w.add(s, t, None, sr_model, i, error=f"{type(e).__name__}: {e}")
        print(f"[omnivoice_clone] {t['id']} done ({w.n_ok} ok / {w.n_fail} failed)", flush=True)
    w.close()


if __name__ == "__main__":
    main()
