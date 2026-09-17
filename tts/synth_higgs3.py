#!/usr/bin/env python3
"""Synthesise lexicon phrases with Higgs Audio v3 (4B) — the model with 100-language support.

Runs in its OWN venv (`.venv_higgs3`): the port needs `torchaudio`, which the benchmark venv
does not have, and the neucodec stack there needs a torchao pin that conflicts.

`bosonai/higgs-tts-3-4b` reports model_type `higgs_multimodal_qwen3`, unsupported by vanilla
transformers — Boson points at SGLang-Omni. `multimodalart/higgs-audio-v3-tts-4b-transformers`
ships the SAME weights (byte-identical, 927 tensors) plus a modeling file, so
`trust_remote_code=True` loads it without SGLang.

Two things the load report makes look alarming but are not:
  * `audio_head.weight MISSING` — it is a tied weight (`_tied_weights_keys`), tied to
    `audio_embedding.weight` at construction. Not randomly initialised in practice.
  * no `.generate()` — audio generation is `model.generate_speech(...)`, which returns a
    mono 24 kHz float32 waveform directly, no manual codebook handling.
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"
SR_OUT = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrases", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tts/out/higgs3"))
    ap.add_argument("--model", default=REPO)
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16).to(args.device).eval()
    sr_model = int(getattr(model.config, "sample_rate", 24000))
    print(f"loaded; codec sample_rate={sr_model}", flush=True)

    wav_dir = args.out / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in args.phrases.open(encoding="utf-8")]
    rows, t0 = [], time.time()

    for i, it in enumerate(items):
        stem = f"{it['lang']}_{i:04d}"
        try:
            with torch.no_grad():
                # Signature is generate_speech(text, tokenizer, *, ...) - text first.
                wav = model.generate_speech(
                    it["phrase"], tok, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                )
            x = np.asarray(wav.detach().cpu().float()).squeeze()
        except Exception as e:
            rows.append({**it, "audio_filepath": "", "duration_s": 0.0,
                         "status": f"error:{type(e).__name__}:{str(e)[:80]}"})
            if i == 0:
                print("first-item error:", type(e).__name__, str(e)[:260], flush=True)
            continue

        if x.size < SR_OUT // 20:
            rows.append({**it, "audio_filepath": "", "duration_s": 0.0, "status": "too_short"})
            continue
        if sr_model != SR_OUT:
            from math import gcd
            g = gcd(sr_model, SR_OUT)
            x = resample_poly(x, SR_OUT // g, sr_model // g)
        peak = float(np.abs(x).max())
        if peak > 0:
            x = np.clip(x / peak * 0.707, -1, 1)
        sf.write(wav_dir / f"{stem}.flac", x.astype("float32"), SR_OUT, format="FLAC", subtype="PCM_16")
        rows.append({**it, "audio_filepath": f"wav/{stem}.flac",
                     "duration_s": round(len(x) / SR_OUT, 3), "status": "ok"})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    man = args.out / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"higgs3: {ok}/{len(rows)} synthesised -> {man}")
    if ok < len(rows):
        from collections import Counter
        print("statuses:", dict(Counter(r["status"][:44] for r in rows)))


if __name__ == "__main__":
    main()
