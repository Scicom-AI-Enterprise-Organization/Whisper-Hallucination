#!/usr/bin/env python3
"""Fetch the audio arms of the corpus. Each arm is independent; run only what you need.

  python scripts/fetch_audio.py aphasia     # 187 wav, 42.8 min, confirmed hallucination triggers
  python scripts/fetch_audio.py halas       # 3,611 Earnings22 segments w/ span-level looping labels
  python scripts/fetch_audio.py musan       # non-speech noise+music (CC BY 4.0, commercial-safe)
  python scripts/fetch_audio.py esc50       # non-speech env. sound (CC BY-NC 3.0, NON-COMMERCIAL)
  python scripts/fetch_audio.py all

Nothing here runs ASR. Transcription is scripts/probe_whisper.py, which is meant to run on
the GPU box, not the laptop.
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def need(tool):
    if not shutil.which(tool):
        sys.exit(f"error: `{tool}` not found on PATH")


def fetch_aphasia():
    """Koenecke et al. FAccT'24 — de-identified AphasiaBank segments that DID hallucinate."""
    dest = AUDIO / "aphasia_koenecke"
    if dest.exists():
        return print(f"[aphasia] already present at {dest.relative_to(ROOT)}")
    need("git")
    tmp = AUDIO / ".tmp_koenecke"
    shutil.rmtree(tmp, ignore_errors=True)
    run(["git", "clone", "--depth", "1", "https://github.com/koenecke/hallucination_harms", str(tmp)])
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp / "data" / "deidentified_hallucinated_audio", dest / "wav", dirs_exist_ok=True)
    shutil.copy(tmp / "data" / "deidentified_transcriptions.csv", dest / "transcriptions.csv")
    shutil.copy(tmp / "README.md", dest / "UPSTREAM_README.md")
    shutil.rmtree(tmp, ignore_errors=True)
    for junk in dest.rglob(".DS_Store"):
        junk.unlink()
    n = len(list((dest / "wav").glob("*.wav")))
    print(f"[aphasia] {n} wav -> {dest.relative_to(ROOT)}")


def fetch_halas():
    """Earnings22 audio backing the HALAS looping annotations."""
    dest = AUDIO / "halas_earnings22"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("error: pip install huggingface_hub")
    print("[halas] downloading distil-whisper/earnings22 (~large; test split only)")
    snapshot_download(
        repo_id="distil-whisper/earnings22",
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["*.json", "*.txt", "*/test*", "README*"],
    )
    print(f"[halas] -> {dest.relative_to(ROOT)}  (labels: manifests/halas_dataset.csv)")


def fetch_musan():
    """MUSAN noise+music. CC BY 4.0 — usable commercially."""
    dest = AUDIO / "musan"
    if (dest / "musan").exists():
        return print(f"[musan] already present at {dest.relative_to(ROOT)}")
    need("curl"); need("tar")
    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / "musan.tar.gz"
    if not tarball.exists():
        print("[musan] downloading ~11 GB from openslr.org/17")
        run(["curl", "-L", "--fail", "-o", str(tarball), "https://www.openslr.org/resources/17/musan.tar.gz"])
    run(["tar", "-xzf", str(tarball), "-C", str(dest)])
    tarball.unlink()
    print(f"[musan] -> {dest.relative_to(ROOT)}  (use noise/ and music/, NOT speech/)")


def fetch_esc50():
    """ESC-50 environmental sound. CC BY-NC 3.0 — NON-COMMERCIAL ONLY."""
    dest = AUDIO / "esc50"
    if dest.exists():
        return print(f"[esc50] already present at {dest.relative_to(ROOT)}")
    need("git")
    print("[esc50] NOTE: CC BY-NC 3.0 — research/eval only, not for commercial training.")
    run(["git", "clone", "--depth", "1", "https://github.com/karoldvl/ESC-50", str(dest)])
    n = len(list((dest / "audio").glob("*.wav")))
    print(f"[esc50] {n} clips -> {dest.relative_to(ROOT)}")


def fetch_urbansound8k():
    """UrbanSound8K. CC BY-NC 4.0 -- NON-COMMERCIAL, eval only, never redistributed.

    Worth fetching despite the licence because it is the benchmark the mitigation papers
    report on, so it is how our numbers become comparable to theirs.
    """
    dest = AUDIO / "urbansound8k"
    if dest.exists():
        return print(f"[urbansound8k] already present at {dest.relative_to(ROOT)}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("error: pip install huggingface_hub")
    print("[urbansound8k] NOTE: CC BY-NC 4.0 -- evaluation only. Do NOT add to the published dataset.")
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="danavery/urbansound8K", repo_type="dataset", local_dir=str(dest))
    print(f"[urbansound8k] -> {dest.relative_to(ROOT)}")


# Licence note: esc50 and urbansound8k are CC BY-NC. They are fetched locally for
# comparability with published results and are deliberately NOT part of the released
# dataset, which stays redistributable.
ARMS = {"aphasia": fetch_aphasia, "halas": fetch_halas, "musan": fetch_musan,
        "esc50": fetch_esc50, "urbansound8k": fetch_urbansound8k}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", choices=[*ARMS, "all"])
    args = ap.parse_args()
    selected = list(ARMS) if "all" in args.arms else args.arms
    for name in selected:
        ARMS[name]()


if __name__ == "__main__":
    main()
