#!/usr/bin/env python3
"""Higgs Audio v3 in the cloning arm: render each phrase in each target speaker's voice.

Unlike Scicom's Multilingual-Expressive (speaker-name conditioning, fixed inventory), v3
takes a reference WAVEFORM, so it can be pointed at the same four targets the conversion
candidates convert to -- which is what makes it comparable with them at all.

`generate_speech(text, tokenizer, *, reference_audio, reference_sample_rate, reference_text)`
-- text first, and the reference transcript materially improves the clone, which is why the
targets carry `ref_short_text`. Runs in `.venv_higgs3`.

Licence note: Higgs forbids using its outputs to train non-Boson speech models, so this is
measured for comparison only and cannot feed the published positives (CLAUDE.md).

    .venv_higgs3/bin/python tts/clone_higgs3.py --device cuda:7
"""
import argparse, sys, traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vc_common import ROOT, SR, Writer, load_sources, load_targets  # noqa: E402

REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_out")
    ap.add_argument("--model", default=REPO)
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16).to(args.device).eval()
    sr_model = int(getattr(model.config, "sample_rate", 24000))

    sources, targets = load_sources(args.sources), load_targets(args.targets)
    if args.limit:
        sources = sources[:args.limit]
    w = Writer(args.out, "higgs3_clone", "higgs-audio-v3", resume=args.resume)
    print(f"[higgs3_clone] {len(sources)} phrases x {len(targets)} targets @ {sr_model} Hz", flush=True)

    for t in targets:
        ref, ref_sr = sf.read(t["ref_short_path"], dtype="float32")
        ref = ref.mean(1) if ref.ndim > 1 else ref
        ref_t = torch.from_numpy(np.ascontiguousarray(ref))
        ref_text = (t.get("ref_short_text") or "").strip()
        if ref_text.isupper():                 # LibriSpeech transcripts are all-caps
            ref_text = ref_text.lower()

        for i, s in enumerate(sources):
            if w.skip(s, t):
                continue
            try:
                with torch.no_grad():
                    wav = model.generate_speech(
                        s["phrase"], tok,
                        reference_audio=ref_t, reference_sample_rate=ref_sr,
                        reference_text=ref_text or None,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_p=args.top_p)
                x = np.asarray(wav.detach().cpu().float()).squeeze()
                if x.size < SR // 20:
                    w.add(s, t, None, sr_model, i, error="output shorter than 50 ms")
                    continue
                w.add(s, t, x, sr_model, i)
            except Exception as e:
                traceback.print_exc()
                w.add(s, t, None, sr_model, i, error=f"{type(e).__name__}: {e}")
        print(f"[higgs3_clone] {t['id']} done ({w.n_ok} ok / {w.n_fail} failed)", flush=True)
    w.close()


if __name__ == "__main__":
    main()
