#!/usr/bin/env python3
"""Synthesise lexicon phrases with Higgs Audio v2 (bosonai/higgs-tts-2-3b-base).

Why v2 and not v3: `higgs-tts-3-4b` reports `model_type: higgs_multimodal_qwen3`, which
transformers 5.17 does not implement -- self-hosting v3 needs SGLang-Omni. v2 is supported
natively (`HiggsAudioV2ForConditionalGeneration`), so it is the version that can actually be
run here and compared like-for-like.

No reference audio is supplied: both models get only text, so the comparison isolates
multilingual intelligibility rather than voice-cloning fidelity. Output is resampled to
16 kHz to match the benchmark arms.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SR_OUT = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrases", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tts/out/higgs"))
    ap.add_argument("--model", default="bosonai/higgs-tts-2-3b-base")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration  # noqa

    torch.manual_seed(args.seed)
    proc = AutoProcessor.from_pretrained(args.model)
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device).eval()
    # The processor keeps its codec on CPU; the codes come off the GPU, so move it.
    at = proc.audio_tokenizer.to(args.device).eval()
    CODEBOOK_SIZE = getattr(at.config, "codebook_size", 1024)
    sr_model = getattr(at.config, "sample_rate", 24000)
    print(f"codec: codebook_size={CODEBOOK_SIZE} sample_rate={sr_model}")

    wav_dir = args.out / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in args.phrases.open(encoding="utf-8")]
    rows, t0 = [], time.time()

    for i, it in enumerate(items):
        stem = f"{it['lang']}_{i:04d}"
        try:
            # The v2 chat template hard-requires a system message (it raises otherwise).
            chat = [
                {"role": "system", "content": "Generate audio following instruction."},
                {"role": "user", "content": it["phrase"]},
            ]
            inputs = proc.apply_chat_template(
                chat, tokenize=True, return_dict=True, add_generation_prompt=True,
                return_tensors="pt",          # the processor accepts nothing else
            )
            inputs = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=args.temperature)

            # generate() yields codebook tokens, not audio. Undo the delay pattern, then
            # drop the stream markers: codebook_size is 1024, and 1024/1025 are
            # audio_stream_bos_id / audio_stream_eos_id. Leaving them in makes the codec's
            # embedding lookup index out of range.
            rev = proc.revert_delay_pattern(out[0])              # (T, K)
            codes = rev.detach().cpu()
            keep = (codes < CODEBOOK_SIZE).all(dim=1)
            if keep.any():                                       # truncate at the first marker frame
                last = int(keep.nonzero()[-1]) + 1
                codes = codes[:last][keep[:last]]
            if codes.numel() == 0:
                rows.append({**it, "audio_filepath": "", "duration_s": 0.0,
                             "status": "no_valid_codes"})
                continue
            with torch.no_grad():
                w = at.decode(codes.T[None].long().to(args.device))   # (B, K, T)
            w = w.audio_values if hasattr(w, "audio_values") else w
            x = np.asarray(w.detach().float().cpu()).squeeze()
        except Exception as e:
            rows.append({**it, "audio_filepath": "", "duration_s": 0.0,
                         "status": f"error:{type(e).__name__}:{str(e)[:90]}"})
            if i == 0:
                print("first-item error:", type(e).__name__, str(e)[:300], flush=True)
            continue

        if x.size < SR_OUT // 20:
            rows.append({**it, "audio_filepath": "", "duration_s": 0.0, "status": "too_short"})
            continue
        if sr_model != SR_OUT:
            from math import gcd
            g = gcd(int(sr_model), SR_OUT)
            x = resample_poly(x, SR_OUT // g, int(sr_model) // g)
        peak = float(np.abs(x).max())
        if peak > 0:
            x = np.clip(x / peak * 0.707, -1, 1)
        sf.write(wav_dir / f"{stem}.flac", x.astype("float32"), SR_OUT, format="FLAC", subtype="PCM_16")
        rows.append({**it, "audio_filepath": f"wav/{stem}.flac",
                     "duration_s": round(len(x) / SR_OUT, 3), "status": "ok"})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    man = args.out / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"higgs: {ok}/{len(rows)} synthesised -> {man}")
    if ok == 0:
        print("statuses:", {r["status"][:40] for r in rows})


if __name__ == "__main__":
    main()
