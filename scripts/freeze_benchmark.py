#!/usr/bin/env python3
"""Freeze the published arms as a BENCHMARK (test-only) and emit the training exclusion list.

The published dataset is an evaluation set, not training data. Declaring every config as
`split: train` invites the one mistake that would invalidate every number we report --
training on the benchmark. This marks it test-only and writes down exactly which source
items are burned, so the training build can exclude them.

Training data must be built SEPARATELY from the same source corpora with these items
removed. `scripts/build_training_corpus.py` consumes the file this writes.

Exclusion identifiers, by stability:
  stable   fsd50k_id, FMA artist+title, LibriSpeech utterance id, Emilia audio_filename
  ordinal  sarawak/manglish/fleurs row index (deterministic for a fixed source revision;
           re-pin the revision if upstream changes)
  n/a      silence, reduplication -- fully synthetic from a seed, no source to exclude

  python scripts/freeze_benchmark.py
"""
import csv, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _disp(p: Path) -> str:
    """Path for display; falls back to the absolute path when it is outside ROOT."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

OUT = ROOT / "benchmark"

# Every published audio arm is evaluation-only.
BENCHMARK_ARMS = ["silence", "music", "nonspeech", "reduplication", "speech_in_noise",
                  "genuine", "genuine_isolated", "librispeech_test_clean"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    excl = defaultdict(set)
    counts = {}

    for arm in BENCHMARK_ARMS:
        man = ROOT / "audio" / arm / "manifest.csv"
        if not man.exists():
            continue
        rows = list(csv.DictReader(man.open(encoding="utf-8")))
        counts[arm] = len(rows)
        for r in rows:
            if arm == "music":
                # FMA: artist+title identifies the track; excerpts of it must not train.
                excl["fma_tracks"].add(f"{r.get('artist','')}|{r.get('title','')}")
            elif arm == "nonspeech":
                excl["fsd50k_ids"].add(r.get("fsd50k_id", ""))
            elif arm in ("genuine", "librispeech_test_clean"):
                excl[f"{r.get('source','')}_stems"].add(Path(r["audio_filepath"]).stem)
            elif arm == "speech_in_noise":
                # Derived from `genuine`; exclude the parent utterance, not the mix.
                excl["genuine_parent_stems"].add(
                    Path(r["audio_filepath"]).stem.rsplit("_", 2)[0])
            elif arm == "genuine_isolated":
                excl["lingualibre_stems"].add(Path(r["audio_filepath"]).stem)

    payload = {
        "_doc": ("Source items consumed by the benchmark. Training data built from these "
                 "same corpora MUST exclude them, or every reported number is contaminated."),
        "benchmark_arms": counts,
        "exclusions": {k: sorted(v) for k, v in excl.items()},
        "counts": {k: len(v) for k, v in excl.items()},
        "synthetic_no_exclusion_needed": ["silence", "reduplication"],
    }
    f = OUT / "exclusions.json"
    f.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("benchmark arms (all test-only):")
    for a, n in counts.items():
        print(f"  {a:<26}{n:>7} clips")
    print("\nexclusion keys written:")
    for k, n in payload["counts"].items():
        print(f"  {k:<26}{n:>7} source items")
    print(f"\n-> {_disp(f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
