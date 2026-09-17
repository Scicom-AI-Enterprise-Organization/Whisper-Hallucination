#!/usr/bin/env python3
"""Synthesise the whole lexicon queue — batched, sharded across GPUs, resumable.

At ~39k clips this is a multi-hour job, so the three properties that matter are not about
audio quality:

  * **Batched.** One-at-a-time generation is what the probe scripts do and it would take a
    day for the English half alone. Scicom is a causal LM, so it batches with LEFT padding;
    OmniVoice batches natively.
  * **Sharded.** `--shard i --num-shards n` splits the queue by index so GPUs 6 and 7 run
    disjoint halves into separate manifests, with no write contention.
  * **Resumable.** The manifest is append-only and re-running skips finished indices, so a
    crash, a preemption or a deliberate stop costs only the batch in flight.

`max_new_tokens` is scaled to the phrase rather than fixed: NeuCodec runs at 50 tokens/s and
speech at roughly 2.5 words/s, so ~20 tokens/word plus slack. A fixed 1024 spends 20 s of
budget on a two-word phrase, which is both wasteful and how the cloning path ran away.

    .venv_bench/bin/python tts/synth_lexicon.py --engine scicom    --shard 0 --num-shards 2 --device cuda:6
    .venv_omni/bin/python  tts/synth_lexicon.py --engine omnivoice --shard 0 --num-shards 1 --device cuda:7
"""
import argparse, csv, json, re, sys, time, traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

TOK = re.compile(r"<\|s_(\d+)\|>")
SR_OUT = 16000
SR_NEUCODEC = 24000
FIELDS = ["idx", "lang", "phrase", "count", "engine", "voice", "audio_filepath",
          "duration_s", "status", "error"]


def tokens_for(phrase: str, floor: int = 120, per_word: int = 22, cap: int = 900) -> int:
    return min(cap, floor + per_word * max(1, len(phrase.split())))


def write_clip(path: Path, x: np.ndarray, sr: int) -> float:
    x = np.asarray(x, dtype="float32").squeeze()
    if sr != SR_OUT:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), SR_OUT)
        x = resample_poly(x, SR_OUT // g, int(sr) // g)
    peak = float(np.abs(x).max() or 0.0)
    if peak > 0:
        x = np.clip(x / peak * 0.707, -1, 1)
    sf.write(path, x.astype("float32"), SR_OUT, format="FLAC", subtype="PCM_16")
    return round(len(x) / SR_OUT, 3)


class ShardWriter:
    def __init__(self, out: Path, engine: str, shard: int):
        self.dir = out / engine
        (self.dir / "wav").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"manifest.shard{shard}.csv"
        self.done = set()
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(encoding="utf-8") as fh:
                self.done = {int(r["idx"]) for r in csv.DictReader(fh) if r.get("idx")}
        new = not self.done
        self.fh = self.path.open("a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.fh, fieldnames=FIELDS)
        if new:
            self.w.writeheader(); self.fh.flush()
        self.ok = self.fail = 0

    def add(self, item, rel="", dur=0.0, error=""):
        self.w.writerow({"idx": item["idx"], "lang": item["lang"], "phrase": item["phrase"],
                         "count": item["count"], "engine": item["engine"],
                         "voice": item.get("voice") or "", "audio_filepath": rel,
                         "duration_s": dur, "status": "ok" if rel else "fail",
                         "error": error[:160]})
        self.fh.flush()
        self.done.add(item["idx"])
        self.ok += bool(rel); self.fail += (not rel)

    def close(self):
        self.fh.close()


def run_scicom(items, args, w):
    import os
    import nf4_shim; nf4_shim.install()          # must precede neucodec (CLAUDE.md)
    import load_neucodec
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.scicom_model)
    tok.padding_side = "left"                     # generation needs the prompt flush right
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.scicom_model, dtype=torch.bfloat16).to(args.device).eval()
    codec = load_neucodec.load(args.codec_repo, token=os.environ.get("HF_TOKEN")).eval().to(args.device)

    t0 = time.time()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        prompts = [f"<|im_start|>{c['voice']}: {c['phrase']}<|speech_start|>" for c in chunk]
        budget = max(tokens_for(c["phrase"]) for c in chunk)
        try:
            inp = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(args.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=budget, do_sample=True,
                                     temperature=args.temperature,
                                     repetition_penalty=args.repetition_penalty,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, inp["input_ids"].shape[1]:]          # generated suffix only
        except Exception as e:
            traceback.print_exc()
            for c in chunk:
                w.add(c, error=f"batch {type(e).__name__}: {e}")
            continue

        for c, seq in zip(chunk, gen):
            try:
                codes = [int(x) for x in TOK.findall(tok.decode(seq, skip_special_tokens=False))]
                if not codes:
                    w.add(c, error="no speech tokens emitted"); continue
                with torch.no_grad():
                    wav = codec.decode_code(torch.tensor(codes)[None, None].to(args.device))
                rel = f"wav/{c['lang']}_{c['idx']:06d}.flac"
                dur = write_clip(w.dir / rel, wav[0, 0].float().cpu().numpy(), SR_NEUCODEC)
                w.add(c, rel=rel, dur=dur)
            except Exception as e:
                w.add(c, error=f"{type(e).__name__}: {e}")

        done = start + len(chunk)
        if done % (args.batch_size * 10) == 0 or done >= len(items):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(items)}  {w.ok} ok  {w.fail} fail  "
                  f"{rate:.2f} clip/s  eta {(len(items)-done)/max(rate,1e-9)/3600:.1f} h", flush=True)


def run_omnivoice(items, args, w):
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    alias_path = ROOT / "tts" / "omnivoice_lang_alias.json"
    alias = json.loads(alias_path.read_text()) if alias_path.exists() else {}
    torch.manual_seed(args.seed)
    model = OmniVoice.from_pretrained(args.omnivoice_model)
    model = model.to(args.device).eval() if hasattr(model, "to") else model
    gcfg = OmniVoiceGenerationConfig(num_step=args.num_step)
    sr_model = int(getattr(model, "sample_rate", 0)
                   or getattr(getattr(model, "config", None), "sample_rate", 0) or 24000)

    def gen(batch):
        with torch.no_grad():
            return model.generate(text=[c["phrase"] for c in batch],
                                  language=[alias.get(c["lang"], c["lang"]) for c in batch],
                                  generation_config=gcfg)

    t0 = time.time()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        try:
            waves = list(gen(chunk))
        except Exception:
            # One bad phrase raises for the whole batch (CLAUDE.md), so fall back per item.
            waves = []
            for c in chunk:
                try:
                    waves.append(gen([c])[0])
                except Exception as e2:
                    waves.append(None)
                    w.add(c, error=f"{type(e2).__name__}: {e2}")
        for c, wav in zip(chunk, waves):
            if wav is None or c["idx"] in w.done:
                continue
            try:
                x = np.asarray(wav, dtype="float32").squeeze()
                if x.size < SR_OUT // 40:
                    w.add(c, error="output shorter than 25 ms"); continue
                rel = f"wav/{c['lang']}_{c['idx']:06d}.flac"
                w.add(c, rel=rel, dur=write_clip(w.dir / rel, x, sr_model))
            except Exception as e:
                w.add(c, error=f"{type(e).__name__}: {e}")

        done = start + len(chunk)
        if done % (args.batch_size * 10) == 0 or done >= len(items):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(items)}  {w.ok} ok  {w.fail} fail  "
                  f"{rate:.2f} clip/s  eta {(len(items)-done)/max(rate,1e-9)/3600:.1f} h", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=ROOT / "tts" / "lexicon_queue.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "lexicon_synth")
    ap.add_argument("--engine", choices=["scicom", "omnivoice"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=1,
                    help="skip phrases observed fewer than N times. 90%% of the lexicon is "
                         "count==1, and at >=5 words those are single transcript fragments "
                         "('0073a this vehicle s got 72 00 miles on it'), not phrases anyone "
                         "says — both engines score CER ~1.0 on them.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sort-window", type=int, default=512,
                    help="sort by phrase length within windows of N; a batch costs the LONGEST "
                         "generation in it, so mixing a 2-word and a 40-word phrase wastes most "
                         "of the budget. Windowed so priority order is broadly preserved.")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--scicom-model", default="Scicom-intl/Multilingual-Expressive-TTS-1.7B")
    ap.add_argument("--codec-repo", default="Scicom-intl/neucodec")
    ap.add_argument("--omnivoice-model", default="k2-fsa/OmniVoice")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine_key = "multilingual-expressive" if args.engine == "scicom" else "omnivoice"
    items = [json.loads(l) for l in args.queue.open(encoding="utf-8")]
    items = [x for x in items if x["engine"] == engine_key
             and x["idx"] % args.num_shards == args.shard
             and x["count"] >= args.min_count]

    w = ShardWriter(args.out, args.engine, args.shard)
    todo = [x for x in items if x["idx"] not in w.done]
    if args.limit:
        todo = todo[:args.limit]
    if args.sort_window > 1:
        windowed = []
        for i in range(0, len(todo), args.sort_window):
            windowed += sorted(todo[i:i + args.sort_window], key=lambda x: len(x["phrase"].split()))
        todo = windowed
    print(f"[{args.engine} shard {args.shard}/{args.num_shards}] {len(items)} assigned, "
          f"{len(w.done)} already done, {len(todo)} to do", flush=True)

    try:
        if todo:
            (run_scicom if args.engine == "scicom" else run_omnivoice)(todo, args, w)
    finally:
        w.close()
        print(f"[{args.engine} shard {args.shard}] {w.ok} ok, {w.fail} failed -> {w.path}", flush=True)


if __name__ == "__main__":
    main()
