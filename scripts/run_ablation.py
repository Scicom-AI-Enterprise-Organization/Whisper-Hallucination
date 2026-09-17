#!/usr/bin/env python3
"""Run the ablation grid over the eval arms. RUNS ON THE GPU BOX, not the laptop.

  python scripts/run_ablation.py --arms silence reduplication --configs baseline no_condition vad_on
  python scripts/run_ablation.py --all

Writes one JSONL per (config, arm) under ablation/results/, and skips work that is
already there, so an interrupted sweep resumes.
"""
import argparse, csv, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = ROOT / "audio"
RESULTS = ROOT / "ablation" / "results"

# Keys in configs.json that steer the runner rather than the decoder.
META = {"_stage", "_why", "_factor", "_doc", "_engine", "model"}


def load_configs(names=None):
    cfgs = {k: v for k, v in json.loads((ROOT / "ablation" / "configs.json").read_text()).items()
            if not k.startswith("_")}
    base = cfgs["baseline"]
    out = {}
    for name in (names or cfgs):
        if name not in cfgs:
            raise SystemExit(f"unknown config {name!r}; have: {', '.join(cfgs)}")
        merged = {**base, **cfgs[name]}          # OFAT entries are deltas off baseline
        out[name] = merged
    return out


def load_arm(arm: str):
    manifest = ARMS / arm / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"no manifest for arm {arm!r} at {manifest} - build or fetch it first")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    for r in rows:
        r["_abs"] = str(ARMS / arm / r["audio_filepath"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["silence", "reduplication"])
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--all", action="store_true", help="every config in configs.json")
    ap.add_argument("--language", default=None, help="force a language hint (default: let it detect)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    configs = load_configs(None if args.all else args.configs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    loaded, model = None, None

    for cname, cfg in configs.items():
        decode = {k: v for k, v in cfg.items() if k not in META}
        if cfg["model"] != loaded:
            print(f"[model] loading {cfg['model']} on {args.device}")
            model = WhisperModel(cfg["model"], device=args.device, compute_type=args.compute_type)
            loaded = cfg["model"]

        for arm in args.arms:
            out = RESULTS / cname / f"{arm}.jsonl"
            if out.exists() and not args.overwrite:
                print(f"[skip] {cname}/{arm} (exists)")
                continue
            rows = load_arm(arm)
            if args.limit:
                rows = rows[: args.limit]
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".jsonl.part")
            t0 = time.time()
            with tmp.open("w", encoding="utf-8") as fh:
                for i, r in enumerate(rows, 1):
                    kw = dict(decode)
                    if args.language:
                        kw["language"] = args.language
                    segs, info = model.transcribe(r["_abs"], **kw)
                    segs = list(segs)
                    rec = {
                        "audio_filepath": r["audio_filepath"],
                        "hyp": "".join(s.text for s in segs).strip(),
                        "n_segments": len(segs),
                        # Needed to score HR the way arXiv:2609.04561 does (after the
                        # built-in no-speech filter) as well as raw. The two differ by an
                        # order of magnitude in the literature.
                        "max_no_speech_prob": max((s.no_speech_prob for s in segs), default=None),
                        "detected_language": info.language,
                        "language_probability": round(info.language_probability, 4),
                        "reference_text": r.get("reference_text", ""),
                        "meta": {k: v for k, v in r.items() if not k.startswith("_")},
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if i % 50 == 0:
                        print(f"  {cname}/{arm} {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
            tmp.rename(out)
            print(f"[done] {cname}/{arm}  {len(rows)} clips in {time.time()-t0:.0f}s -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
