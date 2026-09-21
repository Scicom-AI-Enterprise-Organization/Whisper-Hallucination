#!/usr/bin/env python3
"""Upload the `lexicon_synth` parquet files to the published dataset.

Only this config's files are touched: `upload_folder` with an allow-pattern scoped to
`data/lexicon_synth/`, so the eight benchmark arms and every lookup table are left exactly as
they are. A dataset push is not the place to find out you have replaced an arm.

    .venv_bench/bin/python scripts/push_lexicon_synth.py --build build/lexicon_synth
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "Scicom-intl/Whisper-Hallucination"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=ROOT / "build" / "lexicon_synth")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--config-name", default="lexicon_synth")
    ap.add_argument("--message", default="Add lexicon_synth config (synthetic positives, train/test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import os
    from huggingface_hub import HfApi

    data_dir = args.build / "data" / args.config_name
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"!! no parquet under {data_dir}", file=sys.stderr)
        return 1
    total = sum(f.stat().st_size for f in files) / 1e6
    for f in files:
        print(f"  {f.name:<34} {f.stat().st_size/1e6:>8.1f} MB")
    print(f"  total {total:.1f} MB -> {args.repo}:data/{args.config_name}/")

    if args.dry_run:
        print("--dry-run: nothing uploaded")
        return 0

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    url = api.upload_folder(
        folder_path=str(args.build),
        repo_id=args.repo, repo_type="dataset",
        allow_patterns=[f"data/{args.config_name}/*"],   # never touch the benchmark arms
        commit_message=args.message,
    )
    print(f"pushed -> {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
