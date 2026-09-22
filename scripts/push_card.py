#!/usr/bin/env python3
"""Push DATASET_CARD.md to the Hub as the dataset's README.md -- the card, nothing else.

`build_hf_release.py` stages a whole release directory; this only replaces the card, so
prose fixes do not mean re-uploading 11,852 clips. It is idempotent: the Hub keeps the
commit history, and an unchanged card is a no-op commit.

Runs on the box, where HF_TOKEN lives in .env:

  set -a && . ./.env && set +a
  .venv_bench/bin/python scripts/push_card.py --message "..."
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "Scicom-intl/Whisper-Hallucination"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--card", default=str(ROOT / "DATASET_CARD.md"))
    ap.add_argument("--message", default="Update dataset card")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"],
                    help="model cards push the same way; only the repo type differs")
    ap.add_argument("--dry-run", action="store_true", help="diff against the live card, upload nothing")
    a = ap.parse_args()

    card = Path(a.card)
    if not card.exists():
        print(f"!! {card} missing", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    local = card.read_text()
    try:
        live = Path(hf_hub_download(a.repo, "README.md", repo_type=a.repo_type, token=token)).read_text()
    except Exception as e:                       # first push, or no read access
        live = None
        print(f"could not fetch the live card ({type(e).__name__}); treating as new")

    if live is not None:
        if live == local:
            print("live card already matches DATASET_CARD.md -- nothing to push")
            return 0
        import difflib
        diff = list(difflib.unified_diff(live.splitlines(), local.splitlines(),
                                         "live/README.md", "DATASET_CARD.md", lineterm="", n=1))
        print(f"{sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))} lines added, "
              f"{sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))} removed")
        for line in diff[:200]:
            print(line)

    if a.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    url = api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                          repo_id=a.repo, repo_type=a.repo_type, commit_message=a.message)
    print(f"\npushed -> {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
