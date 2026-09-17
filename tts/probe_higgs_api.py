#!/usr/bin/env python3
"""Find the working audio-decode path for Higgs Audio v2 under transformers 5.17.

`generate()` returns codebook tokens, not a waveform, and the generic
`post_process_multimodal_output` routes to the TEXT decoder. So try the audio paths in
order and report which one produces a plausible waveform.
"""
import inspect

import numpy as np
import torch
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

M = "bosonai/higgs-tts-2-3b-base"
DEV = "cuda:7"

proc = AutoProcessor.from_pretrained(M)
at = proc.audio_tokenizer
print("revert_delay_pattern:", inspect.signature(proc.revert_delay_pattern))
print("save_audio:", inspect.signature(proc.save_audio))
print("audio_tokenizer:", type(at).__name__)
print("  decode-ish:", [m for m in dir(at) if "decode" in m.lower()])
try:
    print("  at.decode sig:", inspect.signature(at.decode))
except Exception:
    pass

model = HiggsAudioV2ForConditionalGeneration.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
chat = [{"role": "system", "content": "Generate audio following instruction."},
        {"role": "user", "content": "Terima kasih banyak."}]
inp = proc.apply_chat_template(chat, tokenize=True, return_dict=True,
                               add_generation_prompt=True, return_tensors="pt")
inp = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inp.items()}
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=768, do_sample=True, temperature=0.8)
print("\ngenerate ->", tuple(out.shape))


def describe(tag, x):
    if x is None:
        return
    a = x
    if torch.is_tensor(a):
        a = a.detach().float().cpu().numpy()
    a = np.asarray(a).squeeze()
    print(f"  [{tag}] shape={a.shape} dtype={a.dtype} "
          f"min={float(a.min()):.4f} max={float(a.max()):.4f}")
    return a


# Path 1: revert the delay pattern, then decode. `revert` returns (T, K); the DAC-style
# decoder expects (B, K, T), so the transpose matters -- feeding (B, T, K) makes the
# codebook embedding lookup index out of range.
rev = proc.revert_delay_pattern(out[0])
print("\nrevert(out[0]) ->", tuple(rev.shape))
cands = [
    ("(B,K,T) = rev.T[None]", rev.T[None]),
    ("(B,K,T) = rev.permute(1,0)[None]", rev.permute(1, 0)[None].contiguous()),
    ("(K,T) = rev.T", rev.T),
    ("(B,T,K) = rev[None]", rev[None]),
]
for tag, arg in cands:
    try:
        with torch.no_grad():
            w = at.decode(arg.to(DEV).long())
        a = describe(f"at.decode {tag} shape_in={tuple(arg.shape)}", w)
        if a is not None and a.size > 1000:
            import soundfile as sf
            sr = getattr(at.config, "sampling_rate", 24000)
            sf.write("/tmp/higgs_probe.wav", a.astype("float32"), sr)
            print(f"  WORKS -> wrote {a.size} samples @ {sr} Hz ({a.size/sr:.2f}s)")
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"  {tag}: {type(e).__name__}: {str(e)[:100]}")

# Path 2: let save_audio do the whole thing.
try:
    proc.save_audio(out, "/tmp/higgs_probe.wav")
    import soundfile as sf
    x, sr = sf.read("/tmp/higgs_probe.wav")
    print(f"\nsave_audio worked: {len(x)} samples @ {sr} Hz")
except Exception as e:
    print(f"\nsave_audio failed: {type(e).__name__}: {str(e)[:140]}")
