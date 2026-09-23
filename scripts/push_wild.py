#!/usr/bin/env python3
"""Upload the `wild` config: real audio that made an ASR model hallucinate or loop.

Scoped exactly like `push_lexicon_synth.py` -- `allow_patterns` on this config's directory and
`delete_patterns` for its stale shards, so the eight benchmark arms and `lexicon_synth` are
untouched and a changed shard count cannot leave two generations behind the same glob.

    set -a && . ./.env && set +a
    .venv_wild/bin/python scripts/push_wild.py --build build/wild
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "Scicom-intl/Whisper-Hallucination"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=ROOT / "build" / "wild")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--config-name", default="wild")
    ap.add_argument("--message", default="Add wild config: real audio that triggers hallucination")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import os
    from huggingface_hub import HfApi

    data_dir = a.build / "data" / a.config_name
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"!! no parquet under {data_dir}", file=sys.stderr)
        return 1
    for f in files:
        print(f"  {f.name:<34} {f.stat().st_size/1e6:>8.1f} MB")
    biggest = max(f.stat().st_size for f in files)
    print(f"  total {sum(f.stat().st_size for f in files)/1e6:.1f} MB -> "
          f"{a.repo}:data/{a.config_name}/")
    if biggest > 300e6:
        print(f"!! largest shard is {biggest/1e6:.0f} MB; the dataset viewer refuses to scan "
              f"past 300 MB", file=sys.stderr)
        return 1
    if a.dry_run:
        print("--dry-run: nothing uploaded")
        return 0

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    url = api.upload_folder(
        folder_path=str(a.build),
        repo_id=a.repo, repo_type="dataset",
        allow_patterns=[f"data/{a.config_name}/*"],
        delete_patterns=[f"data/{a.config_name}/*.parquet"],
        commit_message=a.message,
    )
    print(f"pushed -> {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
