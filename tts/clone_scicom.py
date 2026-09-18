#!/usr/bin/env python3
"""Scicom Multilingual-TTS in the cloning arm — reference audio, not just a speaker name.

Two prompt shapes, and the difference decides which arm this system belongs in:

  --mode clone   zero-shot voice cloning from a reference WAVEFORM, per the base model card
                 (https://huggingface.co/Scicom-intl/Multilingual-TTS-1.7B-Base#voice-cloning):

                   <|im_start|>{ref transcript}<|speech_start|>{ref NeuCodec tokens}<|im_end|>
                   <|im_start|>{text}<|speech_start|>

                 The reference is encoded with the SAME codec that decodes the output, so the
                 speaker arrives as audio tokens rather than as a name.

  --mode named   the fine-tune's speaker-name conditioning, `<|im_start|>{speaker}: {text}`,
                 which cannot aim at a given target and is scored as untargeted.

    .venv_bench/bin/python tts/clone_scicom.py --mode clone --device cuda:6
"""
import argparse, re, sys, traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vc_common import ROOT, Writer, load_sources, load_targets  # noqa: E402

TOK = re.compile(r"<\|s_(\d+)\|>")
SR_CODEC = 24000
SR_REF = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["clone", "named"], default="clone")
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_out")
    ap.add_argument("--model", default="Scicom-intl/Multilingual-Expressive-TTS-1.7B")
    ap.add_argument("--codec-repo", default="Scicom-intl/neucodec")
    ap.add_argument("--system-name", default="")
    ap.add_argument("--speakers", nargs="+",
                    default=["multilingual-tts_audio_Grace", "multilingual-tts_audio_Ryan",
                             "DisfluencySpeech", "multilingual-tts_audio_Serena"],
                    help="--mode named only; names must exist VERBATIM in the `speaker` column of "
                         "Scicom-intl/ExpressiveSpeech (config `data`) -- an unknown name is not an "
                         "error, it just conditions on a token the fine-tune never saw. "
                         "`multilingual-tts_audio_Rahman` was exactly that mistake: the only Rahman "
                         "in the dataset is `genshin-voice_audio_Rahman` (46 rows of Japanese), so "
                         "the slot is now Ryan -- male, Southeast Asian, 587 rows, same studio "
                         "family as Grace/Serena. Four of them so this grid matches the four "
                         "reference targets.")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--greedy", action="store_true",
                    help="do_sample=False; sampling over-generates badly on 2-word targets")
    ap.add_argument("--resume", action="store_true")
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

    if args.mode == "clone":
        targets = load_targets(args.targets)
        system = args.system_name or "scicom_clone"
    else:
        targets = [{"id": spk, "lang": "n/a"} for spk in args.speakers]
        system = args.system_name or "scicom_untargeted"

    w = Writer(args.out, system, args.sources.name, resume=args.resume)
    print(f"[{system}] {len(sources)} phrases x {len(targets)} targets ({args.mode})", flush=True)

    # Encode each reference ONCE: the codec pass is the expensive part of the prompt.
    ref_prefix = {}
    if args.mode == "clone":
        for t in targets:
            y, sr = sf.read(t["ref_short_path"], dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            if sr != SR_REF:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SR_REF)
            with torch.no_grad():
                codes = codec.encode_code(torch.tensor(y)[None, None].to(args.device))
            toks = "".join(f"<|s_{int(i)}|>" for i in codes[0, 0])
            rt = (t.get("ref_short_text") or "").strip()
            if rt.isupper():                 # LibriSpeech transcripts are all-caps
                rt = rt.lower()
            ref_prefix[t["id"]] = f"<|im_start|>{rt}<|speech_start|>{toks}<|im_end|>"
            print(f"  {t['id']}: {codes.shape[-1]} reference codes", flush=True)

    for t in targets:
        for i, s in enumerate(sources):
            if w.skip(s, t):
                continue
            try:
                if args.mode == "clone":
                    prompt = f"{ref_prefix[t['id']]}<|im_start|>{s['phrase']}<|speech_start|>"
                else:
                    prompt = f"<|im_start|>{t['id']}: {s['phrase']}<|speech_start|>"
                inp = tok(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
                with torch.no_grad():
                    gen_kw = dict(max_new_tokens=args.max_new_tokens,
                                  repetition_penalty=args.repetition_penalty)
                    if not args.greedy:
                        gen_kw.update(do_sample=True, temperature=args.temperature)
                    out = model.generate(**inp, **gen_kw)
                dec = tok.decode(out[0], skip_special_tokens=False)
                codes = [int(x) for x in TOK.findall(dec.split("<|speech_start|>")[-1])]
                if not codes:
                    w.add(s, t, None, SR_CODEC, i, error="no speech tokens emitted")
                    continue
                with torch.no_grad():
                    wav = codec.decode_code(torch.tensor(codes)[None, None].to(args.device))
                w.add(s, t, wav[0, 0].float().cpu().numpy(), SR_CODEC, i)
            except Exception as e:
                traceback.print_exc()
                w.add(s, t, None, SR_CODEC, i, error=f"{type(e).__name__}: {e}")
        print(f"[{system}] {t['id']} done ({w.n_ok} ok / {w.n_fail} failed)", flush=True)
    w.close()


if __name__ == "__main__":
    main()
