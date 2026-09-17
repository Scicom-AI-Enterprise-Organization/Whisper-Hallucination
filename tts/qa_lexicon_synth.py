#!/usr/bin/env python3
"""Sample the synthesised lexicon and check it is actually usable as positive examples.

Producing 39k clips proves nothing on its own. A positive example is only a positive if an
ASR can recover the phrase from it, so this samples the manifests and reports, per engine and
per language:

  cer          ASR round-trip against the intended phrase, language forced
  dur_ratio    clip length ÷ expected length (words ÷ 2.5 w/s) — catches the runaway mode
               that made Scicom's *cloning* path unusable, in case it appears here too
  empty_rate   clips the ASR transcribes as nothing at all

Sampling is stratified by language so the 29,884 English clips cannot hide a language that
is entirely broken.

    .venv_bench/bin/python tts/qa_lexicon_synth.py --per-lang 12 --device cuda:6
"""
import argparse, csv, json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer, normalise  # noqa: E402

SR = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path, default=ROOT / "audio" / "lexicon_synth")
    ap.add_argument("--per-lang", type=int, default=12, help="clips sampled per (engine, language)")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "lexicon_synth_qa.json")
    args = ap.parse_args()

    rows = []
    for man in sorted(args.synth.glob("*/manifest.shard*.csv")):
        engine = man.parent.name
        with man.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "ok" and r.get("audio_filepath"):
                    r["_engine"], r["_dir"] = engine, man.parent
                    rows.append(r)
    if not rows:
        print("no finished clips yet"); return

    rng = random.Random(args.seed)
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["_engine"], r["lang"])].append(r)
    sample = []
    for key, items in sorted(by_key.items()):
        sample += rng.sample(items, min(args.per_lang, len(items)))
    # Group by language so a batch never mixes languages (the forced-language decode above
    # applies to the whole batch).
    sample.sort(key=lambda r: (r["_engine"], r["lang"]))
    print(f"{len(rows)} finished clips; sampling {len(sample)} across {len(by_key)} (engine, language) pairs",
          flush=True)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(args.model)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()

    per = defaultdict(lambda: {"cer": [], "dur_ratio": [], "empty": 0, "n": 0})
    for start in range(0, len(sample), args.batch_size):
        chunk = sample[start:start + args.batch_size]
        waves = []
        for r in chunk:
            x, sr = sf.read(r["_dir"] / r["audio_filepath"], dtype="float32")
            waves.append(x.mean(1) if x.ndim > 1 else x)
        # Force the language, exactly as score_tts.py / score_vc.py do. Letting Whisper
        # auto-detect on a short clip in a non-Latin script sends it to the wrong language
        # and returns a CER above 1.0 that says nothing about the audio. Batches are grouped
        # by language for that reason.
        feats = proc(waves, sampling_rate=SR, return_tensors="pt")
        lang = chunk[0]["lang"]
        with torch.no_grad():
            gen = asr.generate(feats.input_features.to(args.device, torch.float16),
                               language=lang if len(lang) == 2 else None,
                               task="transcribe", max_new_tokens=200, num_beams=1)
        hyps = proc.batch_decode(gen, skip_special_tokens=True)
        for r, hyp in zip(chunk, hyps):
            k = (r["_engine"], r["lang"])
            expected = max(0.4, len(r["phrase"].split()) / 2.5)
            per[k]["cer"].append(cer(r["phrase"], hyp.strip()))
            per[k]["dur_ratio"].append(float(r["duration_s"]) / expected)
            per[k]["empty"] += (not normalise(hyp))
            per[k]["n"] += 1
        if (start + len(chunk)) % (args.batch_size * 10) == 0:
            print(f"  scored {start + len(chunk)}/{len(sample)}", flush=True)

    out = {}
    for (engine, lang), v in sorted(per.items()):
        out.setdefault(engine, {})[lang] = {
            "n": v["n"],
            "cer": round(statistics.mean(v["cer"]), 4),
            "median_cer": round(statistics.median(v["cer"]), 4),
            "median_dur_ratio": round(statistics.median(v["dur_ratio"]), 3),
            "empty_rate": round(v["empty"] / v["n"], 4),
        }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    for engine, langs in out.items():
        cers = [x["cer"] for x in langs.values()]
        print(f"\n{engine}: {len(langs)} languages, mean CER {statistics.mean(cers):.3f}, "
              f"median {statistics.median(cers):.3f}")
        worst = sorted(langs.items(), key=lambda kv: -kv[1]["cer"])[:8]
        print("  worst languages:", [(l, v["cer"]) for l, v in worst])
        bad_dur = [(l, v["median_dur_ratio"]) for l, v in langs.items() if v["median_dur_ratio"] > 2]
        print("  over-generating (>2× expected):", bad_dur or "none")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
