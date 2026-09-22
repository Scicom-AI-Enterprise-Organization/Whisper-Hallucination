#!/usr/bin/env python3
"""Measure how many distinct VOICES the synthetic lexicon actually contains.

The manifest's `voice` column is a label, not a measurement: OmniVoice is conditioned on a
language id with no speaker argument at all, so every one of its clips carries `voice=""`.
Whether those clips are one voice per language or a different speaker each time is an
empirical question, and this answers it with the same judge the VC table uses —
WavLM-base-plus-sv x-vectors, cosine on the calibrated scale from
`tts/vc_sim_calibration.json` (≈0.605 between different speakers, ≈0.853 between two clips of
the same one, both measured at 1.6 s).

Per (engine, language) it samples clips and reports the median pairwise cosine:

  near the CEILING  → one voice wearing many filenames
  near the FLOOR    → genuinely different speakers

For the named-speaker engine it splits the pairs into same-name and different-name, which
doubles as a sanity check: if same-name pairs do not score higher than different-name ones,
the conditioning is not reaching the model (the `Rahman` failure mode).

Runs on the box, where the audio lives:

    .venv_omni/bin/python tts/voice_diversity.py --root audio/lexicon_synth_v3 \
        --out tts/voice_diversity.json
"""
import argparse, csv, itertools, json, random, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SV_MODEL = "microsoft/wavlm-base-plus-sv"
SR = 16000


def read16k(path, max_seconds):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    # Cosine rises with clip length, so every comparison is made on the same duration as the
    # calibration (1.6 s) -- otherwise these numbers are not on the same scale as the VC table.
    n = int(max_seconds * SR)
    return x[:n] if x.size >= n else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, nargs="+", default=[Path("audio/lexicon_synth_v3")],
                    help="one or more synthesis roots; clips from all of them are pooled per "
                         "language, which is how the shipped corpus is actually heard")
    ap.add_argument("--engines", nargs="+", default=["scicom", "omnivoice"])
    ap.add_argument("--pool-engines", action="store_true",
                    help="measure all engines together as one pool per language. That is what "
                         "the shipped corpus is -- a language topped up with named speakers "
                         "holds its auto-mode clips AND the named ones -- whereas the default "
                         "per-engine view answers how diverse each generator is on its own.")
    ap.add_argument("--out", type=Path, default=Path("tts/voice_diversity.json"))
    ap.add_argument("--per-lang", type=int, default=24, help="clips sampled per engine/language")
    ap.add_argument("--langs", nargs="+", default=["all"],
                    help='language codes, or "all" for every language with enough clips')
    ap.add_argument("--match-seconds", type=float, default=1.6)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--collapsed-at", type=float, default=0.75,
                    help="median cosine at or above which a language is called one voice")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from transformers import AutoFeatureExtractor, WavLMForXVector
    feat = AutoFeatureExtractor.from_pretrained(SV_MODEL)
    sv = WavLMForXVector.from_pretrained(SV_MODEL).to(args.device).eval()

    def xvec(path):
        x = read16k(path, args.match_seconds)
        inp = feat([x], sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = sv(**{k: v.to(args.device) for k, v in inp.items()}).embeddings
        return torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()

    report = {"match_seconds": args.match_seconds, "per_lang": args.per_lang,
              "roots": [str(r) for r in args.root],
              "pooled": bool(args.pool_engines), "engines": {}}
    groups = [("pooled", args.engines)] if args.pool_engines else [(e, [e]) for e in args.engines]
    for eng, members in groups:
        rows = []
        for root, m_eng in ((r, e) for r in args.root for e in members):
            for m in sorted((root / m_eng).glob("manifest.shard*.csv")):
                for r in csv.DictReader(m.open(encoding="utf-8")):
                    if r["status"] == "ok":
                        # `audio_filepath` is relative to its own root, and `idx` restarts in
                        # each one, so the root -- and the engine directory -- travel with the row.
                        r["_root"] = root
                        r["_eng_dir"] = m_eng
                        rows.append(r)
        by_lang = defaultdict(list)
        for r in rows:
            by_lang[r["lang"]].append(r)

        eng_report = {"n_clips": len(rows),
                      "distinct_voice_labels": len({r["voice"] for r in rows}), "langs": {}}
        langs = sorted(by_lang) if args.langs == ["all"] else args.langs
        for lang in langs:
            pool = by_lang.get(lang, [])
            if len(pool) < 4:
                continue
            sample = rng.sample(pool, min(args.per_lang, len(pool)))
            embs, meta = [], []
            for r in sample:
                p = r["_root"] / r["_eng_dir"] / r["audio_filepath"]
                if not p.exists():
                    continue
                embs.append(xvec(p)); meta.append(r["voice"] or f"<auto:{r['_eng_dir']}>")
            if len(embs) < 4:
                continue
            same, diff, allp = [], [], []
            for i, j in itertools.combinations(range(len(embs)), 2):
                c = float(np.dot(embs[i], embs[j]))
                allp.append(c)
                (same if meta[i] == meta[j] else diff).append(c)
            entry = {"n_clips": len(embs), "n_pairs": len(allp),
                     "median_cosine": round(statistics.median(allp), 4),
                     "p05": round(sorted(allp)[max(0, int(0.05 * len(allp)) - 1)], 4),
                     "p95": round(sorted(allp)[min(len(allp) - 1, int(0.95 * len(allp)))], 4)}
            if same and diff:
                entry["median_same_name"] = round(statistics.median(same), 4)
                entry["median_diff_name"] = round(statistics.median(diff), 4)
                entry["n_same_name_pairs"] = len(same)
            eng_report["langs"][lang] = entry
            print(f"{eng:10s} {lang:3s} n={entry['n_clips']:3d} "
                  f"median={entry['median_cosine']:.3f} "
                  f"[{entry['p05']:.3f}, {entry['p95']:.3f}]"
                  + (f"  same-name={entry['median_same_name']:.3f}"
                     f" diff-name={entry['median_diff_name']:.3f}" if same and diff else ""),
                  flush=True)
        meds = [v["median_cosine"] for v in eng_report["langs"].values()]
        if meds:
            eng_report["median_over_languages"] = round(statistics.median(meds), 4)
            eng_report["languages_near_single_voice"] = sorted(
                l for l, v in eng_report["langs"].items() if v["median_cosine"] >= args.collapsed_at)
        report["engines"][eng] = eng_report

    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
