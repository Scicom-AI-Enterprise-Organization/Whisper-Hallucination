#!/usr/bin/env python3
"""kNN-VC: WavLM features, nearest-neighbour matched against a target speaker, HiFi-GAN out.

Structurally different from the other three: there is no speaker embedding and no training
for the pair -- the target IS a pile of that speaker's WavLM frames, and conversion is a
kNN lookup. That is why it needs `pool/` (minutes) rather than `ref.flac` (one clip), and
why it should be the most language-agnostic of the four: nothing in the path models text.

    .venv_knnvc/bin/python tts/vc_knnvc.py --device cuda:6
"""
import argparse, sys, traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vc_common import ROOT, Writer, load_sources, load_targets, patch_torchaudio_load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_out")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--repo", type=Path, default=ROOT / "vc_repos" / "knn-vc")
    ap.add_argument("--limit", type=int, default=0, help="first N source clips only (smoke test)")
    args = ap.parse_args()

    patch_torchaudio_load()                   # the repo loads audio via torchaudio.load
    sys.path.insert(0, str(args.repo))
    knn_vc = torch.hub.load(str(args.repo), "knn_vc", source="local", prematched=True,
                            trust_repo=True, pretrained=True, device=args.device)

    sources, targets = load_sources(args.sources), load_targets(args.targets)
    if args.limit:
        sources = sources[:args.limit]
    w = Writer(args.out, "knnvc", args.sources.name)
    print(f"[knnvc] {len(sources)} sources x {len(targets)} targets", flush=True)

    for t in targets:
        # The matching set is per target, not per pair -- build it once and reuse it.
        matching = knn_vc.get_matching_set(t["pool_paths"])
        print(f"[knnvc] {t['id']}: matching set {tuple(matching.shape)} from {len(t['pool_paths'])} clips",
              flush=True)
        for i, s in enumerate(sources):
            try:
                q = knn_vc.get_features(s["abs_path"])
                out = knn_vc.match(q, matching, topk=args.topk)
                w.add(s, t, out.cpu().numpy(), 16000, i)
            except Exception as e:
                traceback.print_exc()
                w.add(s, t, None, 16000, i, error=f"{type(e).__name__}: {e}")
    w.close()


if __name__ == "__main__":
    main()
