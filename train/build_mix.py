#!/usr/bin/env python3
"""Build the training mixes for the sweep.

The corpus as staged is **74% blank-label** (24,212 of 32,527 clips): nonspeech, music and
silence whose target is the empty string. Train on that alone and you get exactly the failure
this benchmark measures on `Malaysian-turbo-v3` -- best-in-class on non-speech, and it deletes
46% of real speech. The blank:positive ratio is therefore the knob the sweep exists to turn,
not a detail.

The counterweight is `lexicon_synth`: 22,845 clips in 83 languages where a known hallucination
phrase IS spoken and must be transcribed. It is the contrastive half of the same phrases the
blanks teach the model to stay quiet on.

Mixes:

  blank_only    24,212 blanks, nothing else. The naive recipe, included to reproduce the
                failure rather than assume it.
  corpus        the staged corpus unchanged: 74% blank.
  plus_synth    corpus + every lexicon_synth positive. Blank share falls to ~44%.
  balanced      blanks downsampled to 1:1 against positives.
  synth_heavy   blanks downsampled to 1:2 against positives.

Every mix is disjoint from the benchmark by construction -- the corpus was built that way
(`scripts/verify_disjoint.py`) and `lexicon_synth` train excludes every `targets.csv` phrase.

    python train/build_mix.py --out train/mixes
"""
import argparse, csv, json, random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_corpus(path: Path):
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        rows.append({"audio_filepath": r["audio_filepath"], "text": r.get("text", ""),
                     "lang": r.get("lang") or "", "arm": r.get("arm", ""),
                     "is_blank": bool(r.get("is_blank_label")),
                     "duration_s": float(r.get("duration_s") or 0)})
    return rows


REPO = "Scicom-intl/Whisper-Hallucination"


def load_synth_from_hub(cache: Path):
    """`lexicon_synth` train split, straight from the published dataset.

    Pulled from the Hub rather than the box's build directory so the mix is reproducible by
    anyone: the published split is the one with the benchmark phrases held out. Clips are
    materialised once to `cache` because the sweep reads the mix many times and the trainer
    wants paths, not bytes.
    """
    import io
    import soundfile as sf
    from datasets import Audio, load_dataset

    cache.mkdir(parents=True, exist_ok=True)
    done = cache / ".complete"
    manifest = cache / "manifest.jsonl"
    if done.exists() and manifest.exists():
        rows = [json.loads(l) for l in manifest.open(encoding="utf-8")]
        print(f"lexicon_synth: {len(rows)} clips already materialised at {cache}")
        return rows

    ds = load_dataset(REPO, "lexicon_synth", split="train").cast_column("audio", Audio(decode=False))
    rows = []
    with manifest.open("w", encoding="utf-8") as fh:
        for r in ds:
            path = cache / f"{r['id']}.flac"
            if not path.exists():
                path.write_bytes(r["audio"]["bytes"])
            row = {"audio_filepath": str(path), "text": r["phrase"], "lang": r["lang"],
                   "arm": "lexicon_synth", "is_blank": False,
                   "duration_s": float(r.get("duration_s") or 0)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    done.touch()
    print(f"lexicon_synth: materialised {len(rows)} clips -> {cache}")
    return rows


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    blanks = sum(r["is_blank"] for r in rows)
    hrs = sum(r["duration_s"] for r in rows) / 3600
    print(f"{path.name:<22} {len(rows):>7} clips  {blanks/max(len(rows),1):>5.0%} blank  "
          f"{hrs:>6.1f} h  arms={dict(Counter(r['arm'] for r in rows).most_common(4))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("corpus/train.jsonl"))
    ap.add_argument("--synth-cache", type=Path, default=Path("audio/lexicon_synth_hub"),
                    help="where the published lexicon_synth train clips are materialised")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("train/mixes"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = load_corpus(args.corpus)
    synth = [] if args.no_synth else load_synth_from_hub(args.synth_cache)
    blanks = [r for r in corpus if r["is_blank"]]
    pos = [r for r in corpus if not r["is_blank"]]
    print(f"corpus: {len(blanks)} blank + {len(pos)} positive | lexicon_synth: {len(synth)}")

    def shuffled(xs):
        xs = list(xs); rng.shuffle(xs); return xs

    all_pos = pos + synth
    write(args.out / "blank_only.jsonl", shuffled(blanks))
    write(args.out / "corpus.jsonl", shuffled(corpus))
    write(args.out / "plus_synth.jsonl", shuffled(corpus + synth))
    write(args.out / "balanced.jsonl",
          shuffled(rng.sample(blanks, min(len(blanks), len(all_pos))) + all_pos))
    write(args.out / "synth_heavy.jsonl",
          shuffled(rng.sample(blanks, min(len(blanks), len(all_pos) // 2)) + all_pos))


if __name__ == "__main__":
    main()
