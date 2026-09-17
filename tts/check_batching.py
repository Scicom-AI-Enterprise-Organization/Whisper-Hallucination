#!/usr/bin/env python3
"""Does batched generation produce the same quality as one-at-a-time?

The scaled run batches Scicom with LEFT padding for a 4.7x speedup. If the padding or the
position ids were mishandled the whole 30k-clip run would be quietly degraded, and CER on
the output would be the first place it showed -- so verify it directly rather than trust it.

Same phrases, same seed, both paths, ASR round-trip CER with the language forced.

    .venv_bench/bin/python tts/check_batching.py --n 24 --device cuda:6
"""
import argparse, json, re, statistics, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer  # noqa: E402

TOK = re.compile(r"<\|s_(\d+)\|>")
SR_NEUCODEC, SR_OUT = 24000, 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=ROOT / "tts" / "lexicon_queue.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os
    import nf4_shim; nf4_shim.install()
    import load_neucodec
    from scipy.signal import resample_poly
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              WhisperForConditionalGeneration, WhisperProcessor)

    items = [json.loads(l) for l in args.queue.open(encoding="utf-8")]
    items = [x for x in items if x["lang"] == args.lang and x["engine"] == "multilingual-expressive"][:args.n]
    print(f"{len(items)} {args.lang} phrases", flush=True)

    model_id = "Scicom-intl/Multilingual-Expressive-TTS-1.7B"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(args.device).eval()
    codec = load_neucodec.load("Scicom-intl/neucodec", token=os.environ.get("HF_TOKEN")).eval().to(args.device)

    proc = WhisperProcessor.from_pretrained(args.asr)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.asr, dtype=torch.float16).to(args.device).eval()

    def decode_codes(codes):
        with torch.no_grad():
            wav = codec.decode_code(torch.tensor(codes)[None, None].to(args.device))
        x = wav[0, 0].float().cpu().numpy()
        return resample_poly(x, SR_OUT, SR_NEUCODEC)

    def transcribe(x):
        f = proc(x, sampling_rate=SR_OUT, return_tensors="pt")
        with torch.no_grad():
            g = asr.generate(f.input_features.to(args.device, torch.float16),
                             language=args.lang, task="transcribe", max_new_tokens=200, num_beams=1)
        return proc.batch_decode(g, skip_special_tokens=True)[0].strip()

    prompts = [f"<|im_start|>{c['voice']}: {c['phrase']}<|speech_start|>" for c in items]
    budget = 900
    results = {}

    for mode in ("single", "batched"):
        torch.manual_seed(args.seed)
        tok.padding_side = "left"
        cers, empties = [], 0
        if mode == "single":
            for c, pr in zip(items, prompts):
                inp = tok(pr, return_tensors="pt", add_special_tokens=True).to(args.device)
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=budget, do_sample=True,
                                         temperature=0.8, repetition_penalty=1.15,
                                         pad_token_id=tok.pad_token_id)
                seq = out[0, inp["input_ids"].shape[1]:]
                codes = [int(x) for x in TOK.findall(tok.decode(seq, skip_special_tokens=False))]
                if not codes:
                    empties += 1; cers.append(1.0); continue
                cers.append(cer(c["phrase"], transcribe(decode_codes(codes))))
        else:
            inp = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(args.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=budget, do_sample=True,
                                     temperature=0.8, repetition_penalty=1.15,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, inp["input_ids"].shape[1]:]
            for c, seq in zip(items, gen):
                codes = [int(x) for x in TOK.findall(tok.decode(seq, skip_special_tokens=False))]
                if not codes:
                    empties += 1; cers.append(1.0); continue
                cers.append(cer(c["phrase"], transcribe(decode_codes(codes))))
        results[mode] = {"mean_cer": round(statistics.mean(cers), 4),
                         "median_cer": round(statistics.median(cers), 4),
                         "empty": empties, "n": len(cers)}
        print(f"  {mode:<8} {results[mode]}", flush=True)

    d = results["batched"]["mean_cer"] - results["single"]["mean_cer"]
    print(f"\nbatched − single = {d:+.4f} mean CER")
    print("VERDICT:", "batching is fine" if d < 0.05 else "BATCHING DEGRADES OUTPUT — stop the run")


if __name__ == "__main__":
    main()
