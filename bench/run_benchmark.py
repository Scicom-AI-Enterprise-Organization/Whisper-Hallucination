#!/usr/bin/env python3
"""Run a Whisper checkpoint over the Whisper-Hallucination benchmark.

Pulls the benchmark straight from the Hub, so this is exactly the path anyone reproducing
the numbers would take -- no local prep, no private data.

    python run_benchmark.py --model openai/whisper-large-v3 --device cuda:6

Decoding is deliberately plain: greedy, no forced language, no temperature fallback, no
suppression beyond the model defaults. The point is to measure what the CHECKPOINT does,
not what a decoding wrapper can paper over. Any mitigation is a separate arm.

Language is auto-detected rather than forced, including on the speech arms, because
language drift on low-evidence audio is part of the failure being measured.

Audio is read with soundfile via `Audio(decode=False)`; `datasets>=5` otherwise demands
torchcodec just to hand back an array.
"""
import argparse, io, json, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO = "Scicom-intl/Whisper-Hallucination"
ARMS = ["silence", "music", "nonspeech", "reduplication", "speech_in_noise",
        "genuine", "genuine_isolated", "librispeech_test_clean",
        # Not a benchmark arm -- synthetic positives. Its `test` split is the contrastive
        # other half of `silence`/`music`/`nonspeech`: the SAME phrases, but actually spoken.
        # Emitting the phrase here is the correct answer, so it measures what a hallucination
        # filter would wrongly delete.
        "lexicon_synth"]
SR = 16000


def reference_of(row: dict, arm: str) -> str:
    if arm in ("silence", "music", "nonspeech"):
        return ""                                   # correct output is nothing at all
    if arm == "lexicon_synth":
        return row.get("phrase") or ""              # the phrase the clip actually says
    return row.get("reference_text") or row.get("text") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=440)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pad-lead", type=float, default=0.0,
                    help="seconds of silence prepended to every clip")
    ap.add_argument("--pad-tail", type=float, default=0.0,
                    help="seconds of silence appended to every clip")
    ap.add_argument("--pad-kind", choices=["zeros", "roomtone"], default="zeros",
                    help="`zeros` is digital silence -- note Whisper already zero-pads every "
                         "clip to a 30 s window, so TRAILING zeros change nothing by "
                         "construction and leading zeros only shift where the speech starts. "
                         "`roomtone` pads with real near-silence from the `silence` arm, which "
                         "is a different signal entirely: a noise floor Whisper can hallucinate "
                         "on. Use zeros as the control and roomtone as the realistic case.")
    ap.add_argument("--pad-tag", default="",
                    help="suffix for the results filename, so padded runs do not overwrite the "
                         "unpadded ones (e.g. lexicon_synth__tone2.jsonl)")
    args = ap.parse_args()

    pad_lead = np.zeros(int(args.pad_lead * SR), dtype="float32")
    pad_tail = np.zeros(int(args.pad_tail * SR), dtype="float32")
    tone = None
    if args.pad_kind == "roomtone" and (pad_lead.size or pad_tail.size):
        # Real room tone, not synthesised noise: the `silence` arm is 42 clips over 6 noise
        # floors, and it is the same material the negative arms measure hallucination on.
        sil = load_dataset(REPO, "silence", split="test").cast_column("audio", Audio(decode=False))
        chunks = []
        for r in sil:
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if sr == SR:
                chunks.append(x)
        tone = np.concatenate(chunks) if chunks else None
        print(f"[pad ] room tone pool: {tone.size/SR:.1f}s from {len(chunks)} clips", flush=True)

    rng = np.random.default_rng(0)

    def pad(x):
        if not (pad_lead.size or pad_tail.size):
            return x
        if tone is None:
            return np.concatenate([pad_lead, x, pad_tail])
        def draw(n):
            if n == 0:
                return np.zeros(0, dtype="float32")
            start = int(rng.integers(0, max(1, tone.size - n)))
            seg = tone[start:start + n]
            return np.pad(seg, (0, max(0, n - seg.size)))
        return np.concatenate([draw(pad_lead.size), x, draw(pad_tail.size)]).astype("float32")

    tag = args.model.split("/")[-1]
    torch.set_grad_enabled(False)
    print(f"[load] {args.model} -> {args.device}", flush=True)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()

    for arm in args.arms:
        out = args.out / tag / f"{arm}{args.pad_tag}.jsonl"
        if out.exists() and not args.overwrite:
            print(f"[skip] {tag}/{arm}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(REPO, arm, split="test").cast_column("audio", Audio(decode=False))
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        padding = (f"  +pad {args.pad_lead}s/{args.pad_tail}s {args.pad_kind}"
                   if (args.pad_lead or args.pad_tail) else "")
        print(f"[run ] {tag}/{arm}{args.pad_tag}: {len(ds)} clips{padding}", flush=True)

        t0, tmp = time.time(), out.with_suffix(".jsonl.part")
        with tmp.open("w", encoding="utf-8") as fh:
            for start in range(0, len(ds), args.batch_size):
                rows = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
                waves = []
                for r in rows:
                    x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    assert sr == SR, f"{r['id']} is {sr} Hz"
                    waves.append(pad(x))
                feats = processor(waves, sampling_rate=SR, return_tensors="pt",
                                  return_attention_mask=True)
                inp = feats.input_features.to(args.device, torch.float16)
                am = feats.get("attention_mask")
                gen = model.generate(
                    inp,
                    attention_mask=am.to(args.device) if am is not None else None,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False, num_beams=1,          # greedy: measure the checkpoint
                )
                texts = processor.batch_decode(gen, skip_special_tokens=True)
                for r, hyp in zip(rows, texts):
                    fh.write(json.dumps({
                        "audio_filepath": r.get("id", ""),
                        "arm": arm,
                        "pad": {"lead_s": args.pad_lead, "tail_s": args.pad_tail,
                                "kind": args.pad_kind},
                        "hyp": hyp.strip(),
                        "reference_text": reference_of(r, arm),
                        "meta": {k: v for k, v in r.items() if k != "audio"},
                    }, ensure_ascii=False) + "\n")
                done = min(start + args.batch_size, len(ds))
                if done % (args.batch_size * 20) == 0 or done == len(ds):
                    el = time.time() - t0
                    print(f"   {done}/{len(ds)}  {el:.0f}s  ({done/max(el,1e-9):.1f} clip/s)", flush=True)
        tmp.rename(out)
        print(f"[done] {tag}/{arm} -> {out}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
