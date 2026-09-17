#!/usr/bin/env python3
"""Scicom Multilingual-Expressive-TTS-1.7B in the UNTARGETED arm.

This model cannot join the cloning comparison, and the reason is architectural rather than a
quality gap: it conditions on a speaker NAME token drawn from a fixed inventory
(`Scicom-intl/ExpressiveSpeech`), not on a reference waveform. It can render a phrase in *a*
distinct voice; it cannot render it in *your* target's voice.

So it is measured on what it can actually do:
  * CER / ΔCER, exactly as every other system -- can the ASR still recover the phrase
  * distinctness from the source voice -- does it give the positive pool a different speaker
and its "toward the target" column is empty by construction, not by failure. Each named
speaker fills one target slot so the grid shape matches the rest of the table.

    .venv_bench/bin/python tts/clone_scicom.py --device cuda:6
"""
import argparse, re, sys, traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vc_common import ROOT, Writer, load_sources  # noqa: E402

TOK = re.compile(r"<\|s_(\d+)\|>")
SR_CODEC = 24000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_out")
    ap.add_argument("--model", default="Scicom-intl/Multilingual-Expressive-TTS-1.7B")
    ap.add_argument("--codec-repo", default="Scicom-intl/neucodec")
    ap.add_argument("--speakers", nargs="+",
                    default=["multilingual-tts_audio_Grace", "multilingual-tts_audio_Rahman",
                             "DisfluencySpeech"],
                    help="speaker NAMES; the card recommends multilingual-tts_audio_*")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import os
    import nf4_shim; nf4_shim.install()      # must precede neucodec (CLAUDE.md)
    import load_neucodec
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    codec = load_neucodec.load(args.codec_repo, token=os.environ.get("HF_TOKEN")).eval().to(args.device)

    sources = load_sources(args.sources)
    if args.limit:
        sources = sources[:args.limit]
    # Named speakers stand in for target slots; there is no reference clip to score against.
    targets = [{"id": spk, "lang": "n/a"} for spk in args.speakers]

    w = Writer(args.out, "scicom_untargeted", args.sources.name)
    print(f"[scicom_untargeted] {len(sources)} phrases x {len(targets)} named speakers", flush=True)

    for t in targets:
        for i, s in enumerate(sources):
            try:
                prompt = f"<|im_start|>{t['id']}: {s['phrase']}<|speech_start|>"
                inp = tok(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=True,
                                         temperature=args.temperature,
                                         repetition_penalty=args.repetition_penalty)
                tail = tok.decode(out[0], skip_special_tokens=False).split("<|speech_start|>")
                codes = [int(x) for x in TOK.findall(tail[1])] if len(tail) > 1 else []
                if not codes:
                    w.add(s, t, None, SR_CODEC, i, error="no speech tokens emitted")
                    continue
                with torch.no_grad():
                    wav = codec.decode_code(torch.tensor(codes)[None, None].to(args.device))
                w.add(s, t, wav[0, 0].float().cpu().numpy(), SR_CODEC, i)
            except Exception as e:
                traceback.print_exc()
                w.add(s, t, None, SR_CODEC, i, error=f"{type(e).__name__}: {e}")
        print(f"[scicom_untargeted] {t['id']} done ({w.n_ok} ok / {w.n_fail} failed)", flush=True)
    w.close()


if __name__ == "__main__":
    main()
