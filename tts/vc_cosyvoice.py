#!/usr/bin/env python3
"""CosyVoice 2 -- `inference_vc` for conversion, `inference_zero_shot` for cloned TTS.

Zero-shot cloning is prompted with reference audio AND its transcript, which is why the
targets carry `ref_text`. Its text frontend only knows zh/en/ja/ko/yue, so the cloning arm
will refuse or mangle most of the 23 languages -- again, that limit is the finding, not a
bug to work around.

    .venv_cosyvoice/bin/python tts/vc_cosyvoice.py --mode vc --device cuda:6
"""
import argparse, faulthandler, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vc_common import ROOT, Writer, load_sources, load_targets  # noqa: E402

# CosyVoice2's supported text frontend languages, by our lexicon's ISO-639-1 codes.
CLONE_LANGS = {"zh", "en", "ja", "ko"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["vc", "clone"], default="vc")
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_out")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "vc_repos" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B")
    ap.add_argument("--repo", type=Path, default=ROOT / "vc_repos" / "CosyVoice")
    ap.add_argument("--all-langs", action="store_true",
                    help="clone mode: attempt every language, not just the supported frontend")
    ap.add_argument("--limit", type=int, default=0, help="first N source clips only (smoke test)")
    ap.add_argument("--ref", choices=["short", "long"], default="short",
                    help="one-shot reference for VC mode; cloning always uses the short one")
    ap.add_argument("--system-name", default="",
                    help="output subdirectory; defaults to the mode's usual name")
    ap.add_argument("--chunk", type=int, default=0,
                    help="exit cleanly after N conversions; with --resume, a wrapper loop then "
                         "restarts -- short runs are reliable, long ones are not")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing manifest and skip finished pairs")
    args = ap.parse_args()

    # SIGFPE from the ONNX tokenizer kills the process outright; faulthandler is the only
    # way to learn which Python frame was on the stack when it happened.
    faulthandler.enable(all_threads=True)

    import os
    # Set before torch initialises CUDA: CosyVoice2 hardcodes cuda:0 internally, so the
    # GPU choice has to be made by masking rather than by a device argument.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.split(":")[-1])
    import torch
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "third_party" / "Matcha-TTS"))
    os.chdir(args.repo)
    from cosyvoice.cli.cosyvoice import CosyVoice2

    cv = CosyVoice2(str(args.model), load_jit=False, load_trt=False, fp16=False)
    out_sr = cv.sample_rate

    sources, targets = load_sources(args.sources), load_targets(args.targets)
    if args.limit:
        sources = sources[:args.limit]
    system = args.system_name or ("cosyvoice" if args.mode == "vc" else "cosyvoice_clone")
    w = Writer(args.out, system, args.sources.name if args.mode == "vc" else "cosyvoice2-zeroshot",
               resume=args.resume)
    print(f"[{system}] {len(sources)} phrases x {len(targets)} targets @ {out_sr} Hz", flush=True)

    # This revision's frontend calls load_wav() on whatever it is handed, so prompt and
    # source are PATHS here -- passing the loaded tensors (as older examples do) dies in
    # soundfile with "Invalid file: tensor(...)".
    # Conversion takes the long reference; cloning takes the short one (CosyVoice warns
    # outright when the prompt text dwarfs the text being synthesised).
    key = "ref_path" if (args.mode == "vc" and args.ref == "long") else "ref_short_path"
    prompts = {t["id"]: t[key] for t in targets}
    n_this_run = 0

    for t in targets:
        for i, s in enumerate(sources):
            if w.skip(s, t):
                continue
            if w.key(s, t) in w.poison:
                print(f"[skip] {t['id']} / {s['phrase']!r} crashed the process twice", flush=True)
                w.add(s, t, None, out_sr, i, error="hard crash (SIGFPE) on two attempts")
                continue
            w.attempting(s, t)
            try:
                if args.mode == "clone":
                    if s["lang"] not in CLONE_LANGS and not args.all_langs:
                        w.add(s, t, None, out_sr, i, error=f"frontend has no '{s['lang']}'")
                        continue
                    # LibriSpeech transcripts are all-caps by convention; a shouting prompt
                    # text is not what the frontend expects.
                    ptext = t.get("ref_short_text", "") or ""
                    if ptext.isupper():
                        ptext = ptext.lower()
                    gen = cv.inference_zero_shot(s["phrase"], ptext, prompts[t["id"]], stream=False)
                else:
                    gen = cv.inference_vc(s["abs_path"], prompts[t["id"]], stream=False)
                chunks = [o["tts_speech"] for o in gen]
                wave = torch.cat(chunks, dim=1) if chunks else None
                w.add(s, t, None if wave is None else wave.numpy(), out_sr, i,
                      error="" if chunks else "no audio returned")
                n_this_run += 1
                if args.chunk and n_this_run >= args.chunk:
                    print(f"[cosyvoice] chunk of {args.chunk} done, exiting to restart", flush=True)
                    w.close()
                    return
            except Exception as e:
                traceback.print_exc()
                w.add(s, t, None, out_sr, i, error=f"{type(e).__name__}: {e}")
    w.close()


if __name__ == "__main__":
    main()
