#!/usr/bin/env python3
"""Probe Higgs Audio v3 (4B) via the community transformers port.

`bosonai/higgs-tts-3-4b` reports model_type `higgs_multimodal_qwen3`, which vanilla
transformers does not implement -- Boson's own docs point at SGLang-Omni. But
`multimodalart/higgs-audio-v3-tts-4b-transformers` ships the same weights alongside
`modeling_higgs_multimodal_qwen3.py`, so `trust_remote_code=True` loads it without SGLang.

This establishes the call signature and the audio-decode path before wiring it into the
comparison; v2 needed three separate corrections (system message, return_tensors, stream
markers), so probe first.
"""
import inspect

import numpy as np
import torch

REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"
DEV = "cuda:7"

from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer  # noqa: E402

cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
print("config:", type(cfg).__name__, "| model_type:", cfg.model_type)
for k in ["audio_codebook_size", "audio_num_codebooks", "audio_stream_bos_id",
          "audio_stream_eos_id", "audio_tokenizer_name_or_path", "audio_in_token_idx",
          "audio_out_token_idx"]:
    if hasattr(cfg, k):
        print(f"  cfg.{k} = {getattr(cfg, k)}")

proc = tok = None
try:
    proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
    print("processor:", type(proc).__name__)
    print("  methods:", [m for m in dir(proc) if not m.startswith('_') and callable(getattr(proc, m))][:18])
except Exception as e:
    print("no processor:", type(e).__name__, str(e)[:110])
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    print("tokenizer:", type(tok).__name__)

model = AutoModelForCausalLM.from_pretrained(
    REPO, trust_remote_code=True, dtype=torch.bfloat16).to(DEV).eval()
print("model:", type(model).__name__)
print("  generate sig:", str(inspect.signature(model.generate))[:200])
for attr in ["audio_tokenizer", "audio_decoder", "codec", "decode_audio"]:
    if hasattr(model, attr):
        print(f"  model.{attr}: {type(getattr(model, attr)).__name__}")

text = "Terima kasih kerana menonton."
try:
    chat = [{"role": "system", "content": "Generate audio following instruction."},
            {"role": "user", "content": text}]
    src = proc if proc is not None else tok
    inputs = src.apply_chat_template(chat, tokenize=True, return_dict=True,
                                     add_generation_prompt=True, return_tensors="pt")
    inputs = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inputs.items()}
    print("\ninput keys:", list(inputs))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.8)
    print("generate ->", type(out).__name__,
          tuple(out.shape) if torch.is_tensor(out) else [a for a in dir(out) if not a.startswith('_')][:12])
    if torch.is_tensor(out):
        a = out.detach().cpu().numpy()
        print("  value range:", int(a.min()), "..", int(a.max()))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("generate path failed:", type(e).__name__, str(e)[:200])
