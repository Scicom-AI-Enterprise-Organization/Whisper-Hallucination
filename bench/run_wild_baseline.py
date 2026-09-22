#!/usr/bin/env python3
"""Baseline the checkpoints on WILD audio -- real recordings that made a model hallucinate.

The eight benchmark arms are built stimuli. This runs the same models over audio nobody
constructed: meeting rooms, earnings calls, parliament, and the aphasia clips from Koenecke
et al. Each clip carries a reason it was collected, and the reason determines what counts as
a failure:

  blank_speech   Silero VAD found no speech. Correct output is nothing, so ANY words are
                 invented -- the same contract as the `silence` arm, on real audio.
  loop           the mining model produced a token run >= 6 here. Measures whether a
                 different checkpoint loops on the same audio.
  halas_*        HALAS human span labels (Earnings-22). `halas_hallucination` means at least
                 one annotator marked a hallucinated span for some model on this clip.
  aphasia        Koenecke et al.'s de-identified clips that hallucinated in their study.

Reported per (model, reason): any-output rate, word-output rate, loop rate, and the phrases
emitted, so the numbers line up with `bench/score_benchmark.py` on the synthetic arms.

    python bench/run_wild_baseline.py --model openai/whisper-large-v3 --device cuda:7
"""
import argparse, csv, io, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import in_lexicon, load_lexicon, max_ngram_repeat, normalise  # noqa: E402

SR = 16000


def load_manifests(roots):
    """Every mined/fetched wild manifest, tagged with which collection it came from."""
    rows = []
    for root in roots:
        for man in sorted(Path(root).rglob("manifest.csv")):
            with man.open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if not r.get("audio_filepath"):
                        continue
                    r["_dir"] = man.parent
                    r["collection"] = man.parent.parent.name
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--roots", nargs="+", default=["audio_wild"])
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--out", type=Path, default=Path("bench/wild_results"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=440)
    ap.add_argument("--loop-threshold", type=int, default=6)
    ap.add_argument("--lexicon", type=Path, default=ROOT / "lexicon" / "combined_lexicon.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_manifests(args.roots)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("no wild manifests found -- run scripts/mine_wild_hallucinations.py first")
    tag = args.model.split("/")[-1]
    print(f"[wild-base] {tag}: {len(rows)} clips from "
          f"{len({r['collection'] for r in rows})} collections", flush=True)

    lex = load_lexicon(args.lexicon) if args.lexicon.exists() else set()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(args.model)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{tag}.jsonl"
    recs, t0 = [], time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            waves = []
            for r in chunk:
                x, sr = sf.read(r["_dir"] / r["audio_filepath"], dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                waves.append(x)
            feats = proc(waves, sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                gen = asr.generate(feats.input_features.to(args.device, torch.float16),
                                   task="transcribe", max_new_tokens=args.max_new_tokens,
                                   do_sample=False, num_beams=1)
            for r, hyp in zip(chunk, proc.batch_decode(gen, skip_special_tokens=True)):
                hyp = hyp.strip()
                rec = {"id": r.get("id", ""), "collection": r["collection"],
                       "reasons": r.get("reasons", ""), "hyp": hyp,
                       "any_output": bool(hyp),
                       "word_output": bool(normalise(hyp)),
                       "max_token_run": max_ngram_repeat(hyp, 1),
                       "in_lexicon": bool(lex) and in_lexicon(hyp, lex),
                       "duration_s": float(r.get("duration_s") or 0),
                       "mined_hyp": r.get("hyp", "")}
                recs.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done = start + len(chunk)
            if done % (args.batch_size * 20) == 0 or done >= len(rows):
                el = time.time() - t0
                print(f"   {done}/{len(rows)}  {done/max(el,1e-9):.1f} clip/s", flush=True)

    # Summary per reason: on blank clips silence is the correct answer, so any output is a
    # hallucination and the rate is directly comparable to the `silence` arm.
    by = defaultdict(list)
    for rec in recs:
        for reason in (rec["reasons"] or "unlabelled").split("|"):
            by[reason].append(rec)
    summary = {"model": args.model, "n": len(recs), "by_reason": {}}
    for reason, items in sorted(by.items()):
        n = len(items)
        summary["by_reason"][reason] = {
            "n": n,
            "any_output_rate": round(sum(x["any_output"] for x in items) / n, 4),
            "word_output_rate": round(sum(x["word_output"] for x in items) / n, 4),
            "loop_rate": round(sum(x["max_token_run"] >= args.loop_threshold for x in items) / n, 4),
            "lexicon_rate": round(sum(x["in_lexicon"] for x in items) / n, 4),
            "top_outputs": [{"text": t[:60], "count": c} for t, c in
                            Counter(normalise(x["hyp"]) for x in items if normalise(x["hyp"])).most_common(5)],
        }
    (args.out / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for reason, s in summary["by_reason"].items():
        print(f"  {reason:<18} n={s['n']:<6} any={s['any_output_rate']:.3f} "
              f"words={s['word_output_rate']:.3f} loop={s['loop_rate']:.3f}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
