#!/usr/bin/env python3
"""Fetch the checkpoints the VC candidates expect at the paths they expect.

seed-vc and kNN-VC pull their own weights on first run (HF hub / torch.hub); these two do
not, and both want the files laid out under their repo.

    .venv_bench/bin/python tts/setup/fetch_vc_models.py
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOS = ROOT / "vc_repos"

WANTED = {
    "openvoice": dict(repo_id="myshell-ai/OpenVoiceV2",
                      local_dir=REPOS / "OpenVoice" / "checkpoints_v2",
                      allow_patterns=["converter/*"]),
    "cosyvoice": dict(repo_id="FunAudioLLM/CosyVoice2-0.5B",
                      local_dir=REPOS / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B",
                      allow_patterns=None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(WANTED), default=sorted(WANTED))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    for name in args.only:
        spec = WANTED[name]
        print(f"[fetch] {name}: {spec['repo_id']} -> {spec['local_dir']}", flush=True)
        p = snapshot_download(repo_id=spec["repo_id"], local_dir=str(spec["local_dir"]),
                              allow_patterns=spec["allow_patterns"])
        files = sorted(x.name for x in Path(p).rglob("*") if x.is_file())
        print(f"[ok   ] {name}: {len(files)} files, {sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file())/1e6:.0f} MB")


if __name__ == "__main__":
    main()
