#!/usr/bin/env python3
"""Diagnose the token-value range problem in Higgs v2 audio decoding.

Every tensor layout failed with "index out of range in self", which is an embedding lookup
receiving out-of-range ids -- so the VALUES are wrong, not the shape. `generate()` returns
ids in the LM's combined vocabulary; audio codes occupy an offset window inside it and have
to be shifted back to 0..codebook_size-1 before the codec can embed them.
"""
import numpy as np
import torch
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

M, DEV = "bosonai/higgs-tts-2-3b-base", "cuda:7"
proc = AutoProcessor.from_pretrained(M)
at = proc.audio_tokenizer
cfg = at.config
print("codec config fields:", {k: v for k, v in vars(cfg).items()
                               if isinstance(v, (int, float, str, bool)) and not k.startswith("_")})

model = HiggsAudioV2ForConditionalGeneration.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
mc = model.config
for k in ["audio_in_token_idx", "audio_out_token_idx", "audio_stream_bos_id", "audio_stream_eos_id",
          "audio_codebook_size", "audio_num_codebooks", "text_vocab_size", "vocab_size"]:
    if hasattr(mc, k):
        print(f"  model.config.{k} = {getattr(mc, k)}")

chat = [{"role": "system", "content": "Generate audio following instruction."},
        {"role": "user", "content": "Terima kasih banyak."}]
inp = proc.apply_chat_template(chat, tokenize=True, return_dict=True,
                               add_generation_prompt=True, return_tensors="pt")
inp = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inp.items()}
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=768, do_sample=True, temperature=0.8)
rev = proc.revert_delay_pattern(out[0])
r = rev.detach().cpu().numpy()
print(f"\nreverted {r.shape}  min={r.min()} max={r.max()}")
print("per-codebook min/max:", [(int(r[:, k].min()), int(r[:, k].max())) for k in range(r.shape[1])])

cbs = getattr(mc, "audio_codebook_size", 1024)
print(f"\ncodebook_size={cbs}; values already in range? {r.max() < cbs}")
for name, shifted in [("raw", r), (f"-{cbs}*k stagger", r - np.arange(r.shape[1])[None, :] * cbs)]:
    print(f"  {name}: min={shifted.min()} max={shifted.max()} in_range={shifted.min() >= 0 and shifted.max() < cbs}")
    if shifted.min() >= 0 and shifted.max() < cbs:
        t = torch.tensor(shifted.T[None], dtype=torch.long, device=DEV)
        try:
            with torch.no_grad():
                w = at.decode(t)
            a = (w.audio_values if hasattr(w, "audio_values") else w).detach().float().cpu().numpy().squeeze()
            sr = getattr(cfg, "sampling_rate", 24000)
            print(f"  DECODED: {a.shape} @ {sr} Hz = {a.size/sr:.2f}s  peak={np.abs(a).max():.3f}")
            import soundfile as sf
            sf.write("/tmp/higgs_probe.wav", a.astype("float32"), sr)
            print("  -> /tmp/higgs_probe.wav")
            break
        except Exception as e:
            print(f"  decode failed: {type(e).__name__}: {str(e)[:110]}")
