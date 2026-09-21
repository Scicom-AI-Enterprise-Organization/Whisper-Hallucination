#!/usr/bin/env python3
"""Score every synthesised clip and keep only the ones that work as positive examples.

Routing by measured CER gets the average right; it does not make any individual clip usable.
A TTS that over-generates on a long phrase emits the phrase plus several seconds of invented
speech, and that clip is a bad positive no matter which engine produced it. So the corpus is
defined by what survives this pass, not by what was generated.

Two gates, both of which a clip must pass:

  cer <= --max-cer                 the ASR recovers the phrase (language FORCED, as in every
                                   other scorer here -- auto-detect on a short non-Latin clip
                                   lands in the wrong language and fails a good clip)
  dur_ratio in [min, max]          the clip is about as long as the phrase should take, so
                                   runaway generation is rejected even when CER survives it

Writes accepted.csv / rejected.csv and a per-language yield report, so a language that
produces nothing usable is visible rather than silently thin.

    .venv_bench/bin/python tts/filter_lexicon_synth.py --device cuda:6
"""
import argparse, csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer, normalise  # noqa: E402

SR = 16000
OUT_FIELDS = ["idx", "lang", "phrase", "count", "engine", "voice", "audio_filepath",
              "duration_s", "cer", "dur_ratio", "hyp", "verdict", "meta"]


def expected_seconds(phrase: str) -> float:
    """How long this phrase should take to say.

    `words / 2.5` underestimates short utterances badly -- a one-word `Ačiū.` runs about
    0.8 s with onset and decay, not 0.4 s, so correct clips were being flagged as 3x
    over-generating. An affine fit is far closer at the short end, which is where this
    corpus lives.
    """
    return 0.35 + 0.45 * max(1, len(phrase.split()))


def recovered(ref: str, hyp: str, max_cer: float) -> bool:
    """Did the ASR recover the phrase?

    CER alone is unusable on very short references: `Pag` against `Pagrindiniai.` scores
    3.00, and a single hallucinated token against a 5-character reference scores above 1.
    For short phrases the question that matters is whether the phrase is THERE, so
    containment stands in; the duration gate separately rejects a clip that buried the
    phrase in seconds of invented speech.
    """
    r, h = normalise(ref), normalise(hyp)
    if not r or not h:
        return False
    if cer(ref, hyp) <= max_cer:
        return True
    return len(r) <= 12 and r in h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path, default=ROOT / "audio" / "lexicon_synth")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-cer", type=float, default=0.25)
    ap.add_argument("--judge-floor", type=Path, default=ROOT / "tts" / "judge_floor.json",
                    help="per-language CER of the judge on REAL speech (tts/judge_floor.py)")
    ap.add_argument("--floor-slack", type=float, default=1.5,
                    help="a language's gate is max(--max-cer, floor * slack)")
    ap.add_argument("--unvalidatable-floor", type=float, default=0.5,
                    help="above this the judge fails on real speech too, so no clip can pass "
                         "honestly; those languages are flagged, not silently dropped")
    ap.add_argument("--min-dur-ratio", type=float, default=0.5)
    ap.add_argument("--max-dur-ratio", type=float, default=1.8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "lexicon_synth")
    args = ap.parse_args()

    # A flat gate is wrong in two opposite ways. Whisper reads Lithuanian at CER 0.057 and
    # Amharic at 1.226 -- on REAL human speech. Holding both to 0.25 fails every Amharic clip
    # regardless of quality, and lets nothing about the judge show up in the report.
    floors = {}
    if args.judge_floor.exists():
        floors = {k: v["median_cer"] for k, v in json.loads(args.judge_floor.read_text()).items()}
        print(f"judge floor known for {len(floors)} languages; "
              f"unvalidatable (> {args.unvalidatable_floor}): "
              f"{sorted(l for l, f in floors.items() if f > args.unvalidatable_floor)}")

    def gate_for(lang):
        if lang not in WHISPER_LANGS:
            return args.max_cer, True          # the judge has no such language
        f = floors.get(lang)
        if f is None:
            return args.max_cer, False
        if f > args.unvalidatable_floor:
            return args.max_cer, True          # flagged: the judge cannot read this language
        return max(args.max_cer, f * args.floor_slack), False

    rows = []
    for man in sorted(args.synth.glob("*/manifest.shard*.csv")):
        with man.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "ok" and r.get("audio_filepath"):
                    r["_dir"] = man.parent
                    rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("nothing to score"); return
    rows.sort(key=lambda r: (r["lang"], r["engine"]))     # forced decode is per batch
    print(f"scoring {len(rows)} clips", flush=True)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

    # A third category, separate from "judge reads it well" and "judge reads it badly":
    # languages Whisper does not have at all (ga, mi-less locales, most of NLLB's tail).
    # Forcing one raises, which killed the first run at row 3,421. These are unvalidatable
    # by definition -- there is no judge to ask.
    WHISPER_LANGS = set(TO_LANGUAGE_CODE.values()) | set(TO_LANGUAGE_CODE)
    proc = WhisperProcessor.from_pretrained(args.model)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()

    acc_fh = (args.out / "accepted.csv").open("w", newline="", encoding="utf-8")
    rej_fh = (args.out / "rejected.csv").open("w", newline="", encoding="utf-8")
    acc = csv.DictWriter(acc_fh, fieldnames=OUT_FIELDS); acc.writeheader()
    rej = csv.DictWriter(rej_fh, fieldnames=OUT_FIELDS); rej.writeheader()

    stats = defaultdict(lambda: {"n": 0, "kept": 0, "cer": [], "reasons": defaultdict(int)})
    batch = []

    def flush(batch):
        if not batch:
            return
        lang = batch[0]["lang"]
        _, unreadable = gate_for(lang)
        if unreadable and lang not in WHISPER_LANGS:
            # No judge for this language; record the clips rather than crash or fake a score.
            for r in batch:
                out = {k: r.get(k, "") for k in OUT_FIELDS}
                out.update(cer="", dur_ratio=round(float(r["duration_s"]) /
                                                   expected_seconds(r["phrase"]), 3),
                           hyp="", verdict="reject:no_judge_for_language")
                rej.writerow(out)
                st = stats[r["lang"]]
                st["n"] += 1; st["reasons"]["no_judge_for_language"] += 1
            return

        waves = []
        for r in batch:
            x, sr = sf.read(r["_dir"] / r["audio_filepath"], dtype="float32")
            waves.append(x.mean(1) if x.ndim > 1 else x)
        feats = proc(waves, sampling_rate=SR, return_tensors="pt")
        try:
            with torch.no_grad():
                gen = asr.generate(feats.input_features.to(args.device, torch.float16),
                                   language=lang if len(lang) == 2 else None,
                                   task="transcribe", max_new_tokens=220, num_beams=1)
        except Exception as e:
            # One unusable batch must not cost the whole pass.
            print(f"  [batch {lang} failed: {type(e).__name__}: {str(e)[:70]}]", flush=True)
            for r in batch:
                out = {k: r.get(k, "") for k in OUT_FIELDS}
                out.update(cer="", dur_ratio="", hyp="", verdict="reject:asr_error")
                rej.writerow(out)
                st = stats[r["lang"]]
                st["n"] += 1; st["reasons"]["asr_error"] += 1
            return
        for r, hyp in zip(batch, proc.batch_decode(gen, skip_special_tokens=True)):
            hyp = hyp.strip()
            c = cer(r["phrase"], hyp)
            ratio = float(r["duration_s"]) / expected_seconds(r["phrase"])
            gate, unvalidatable = gate_for(r["lang"])
            reason = ""
            if unvalidatable:
                reason = "unvalidatable_language"
            elif not recovered(r["phrase"], hyp, gate):
                reason = "cer"
            elif not (args.min_dur_ratio <= ratio <= args.max_dur_ratio):
                reason = "duration"
            elif not normalise(hyp):
                reason = "empty"
            out = {k: r.get(k, "") for k in OUT_FIELDS}
            out.update(cer=round(c, 4), dur_ratio=round(ratio, 3), hyp=hyp[:200],
                       verdict="accept" if not reason else f"reject:{reason}")
            (acc if not reason else rej).writerow(out)
            s = stats[r["lang"]]
            s["n"] += 1; s["cer"].append(c)
            if reason:
                s["reasons"][reason] += 1
            else:
                s["kept"] += 1

    for i, r in enumerate(rows):
        if batch and (r["lang"] != batch[0]["lang"] or len(batch) >= args.batch_size):
            flush(batch); batch = []
        batch.append(r)
        if i and i % (args.batch_size * 40) == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    flush(batch)
    acc_fh.close(); rej_fh.close()

    report = {lang: {"n": s["n"], "kept": s["kept"],
                     "yield": round(s["kept"] / s["n"], 4) if s["n"] else 0.0,
                     "median_cer": round(statistics.median(s["cer"]), 4) if s["cer"] else None,
                     "judge_floor": floors.get(lang),
                     "gate": round(gate_for(lang)[0], 4),
                     "judge_cannot_read": gate_for(lang)[1],
                     "rejects": dict(s["reasons"])}
              for lang, s in sorted(stats.items())}
    (args.out / "filter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    kept = sum(v["kept"] for v in report.values())
    n = sum(v["n"] for v in report.values())
    unval = [l for l, v in report.items() if v["judge_cannot_read"]]
    validatable = {l: v for l, v in report.items() if not v["judge_cannot_read"]}
    vk = sum(v["kept"] for v in validatable.values())
    vn = sum(v["n"] for v in validatable.values())
    print(f"\nkept {kept}/{n} ({kept/max(n,1):.1%}) across {len(report)} languages")
    print(f"  excluding languages the judge cannot read: {vk}/{vn} ({vk/max(vn,1):.1%})")
    if unval:
        print(f"  judge cannot read (flagged, not quality-checked): {unval}")
    worst = sorted(report.items(), key=lambda kv: kv[1]["yield"])[:10]
    print("lowest-yield languages:", [(l, f"{v['yield']:.0%}", v["n"]) for l, v in worst])
    print(f"-> {args.out}/accepted.csv, rejected.csv, filter_report.json")


if __name__ == "__main__":
    main()
