#!/usr/bin/env python3
"""Build the two candidate queues for making the OmniVoice half of the corpus speaker-diverse.

`tts/voice_diversity.py` measured what the manifest cannot say: OmniVoice takes no speaker
argument, and in 33 of its 74 languages the clips sit at a median pairwise cosine of 0.75+ --
one voice, against a calibrated 0.605 for different speakers and 0.853 for the same one. Two
ways out, and they trade against each other, so they get probed on the same phrases:

  scicom  Multilingual-Expressive in speaker-name mode. Measured at ΔCER −0.035 with real
          speaker separation (same-name 0.826 / different-name 0.601), but only over the 21
          languages the TTS ablation covered -- its coverage is untagged, so in a language
          like `mk` or `te` it is an open question.
  clone   OmniVoice conditioned on a reference clip. Apache-2.0, same engine that already
          covers these languages, strongest identity transfer measured (107%) -- and +0.339
          CER over its own auto-mode clip, which the round-trip filter will charge for.

Phrases are drawn from the clips already generated, so both routes are compared against an
existing auto-mode rendering of the same text rather than against each other only.

    python tts/build_diversity_probe.py --collapsed-at 0.75 --langs 4 --phrases 24
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diversity", type=Path, default=ROOT / "tts" / "voice_diversity.json")
    ap.add_argument("--corpus", type=Path, default=Path("audio/lexicon_synth_v3"))
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--speakers", type=Path, default=ROOT / "tts" / "expressive_speakers.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tts")
    ap.add_argument("--collapsed-at", type=float, default=0.75)
    ap.add_argument("--langs", type=int, default=4, help="how many collapsed languages to probe")
    ap.add_argument("--phrases", type=int, default=24, help="phrases per language")
    ap.add_argument("--voices", type=int, default=4, help="voices per phrase, each route")
    ap.add_argument("--named-voices", nargs="+",
                    default=["multilingual-tts_audio_Grace", "multilingual-tts_audio_Ryan",
                             "DisfluencySpeech", "multilingual-tts_audio_Serena"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    known = set(json.loads(args.speakers.read_text())["all_speakers"])
    bad = [v for v in args.named_voices if v not in known]
    if bad:                      # the Rahman failure mode: an unknown name does not raise
        raise SystemExit(f"speaker name(s) not in the inventory, refusing to queue them: {bad}")

    div = json.loads(args.diversity.read_text())["engines"]["omnivoice"]["langs"]
    collapsed = sorted(((v["median_cosine"], l) for l, v in div.items()
                        if v["median_cosine"] >= args.collapsed_at), reverse=True)
    if not collapsed:
        raise SystemExit("no language is above the collapse threshold -- nothing to probe")
    # Spread the probe over the range rather than taking only the worst: the question is
    # whether a route helps at all, not how it does on the single most degenerate language.
    step = max(1, len(collapsed) // args.langs)
    picked = [l for _, l in collapsed[::step]][:args.langs]
    print("collapsed languages probed:",
          ", ".join(f"{l} ({dict((b,a) for a,b in collapsed)[l]:.2f})" for l in picked))

    rows = list(csv.DictReader((args.corpus / "omnivoice" / "manifest.shard0.csv").open(encoding="utf-8")))
    by_lang = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            by_lang[r["lang"]].append(r)

    targets = json.loads((args.targets / "targets.json").read_text())
    refs = []
    for t in targets[:args.voices]:
        refs.append({
            "id": t["id"],
            "ref_audio": str((args.targets / t["id"] / "ref_short.flac").resolve()),
            # LibriSpeech transcripts are all-caps; OmniVoice reads them as shouting.
            "ref_text": (t.get("ref_short_text") or "").strip().lower(),
        })
    if len(refs) < args.voices:
        raise SystemExit(f"only {len(refs)} reference speakers in {args.targets}")

    named, cloned, idx = [], [], 0
    for lang in picked:
        pool = by_lang.get(lang, [])
        if len(pool) < args.phrases:
            print(f"  {lang}: only {len(pool)} clips, using all of them")
        for r in rng.sample(pool, min(args.phrases, len(pool))):
            for v in range(args.voices):
                named.append({"idx": idx, "phrase": r["phrase"], "lang": lang,
                              "count": int(r["count"]), "source": "diversity_probe",
                              "engine": "multilingual-expressive",
                              "voice": args.named_voices[v % len(args.named_voices)]})
                ref = refs[v % len(refs)]
                cloned.append({"idx": idx, "phrase": r["phrase"], "lang": lang,
                               "count": int(r["count"]), "source": "diversity_probe",
                               "engine": "omnivoice", "voice": ref["id"],
                               "ref_audio": ref["ref_audio"], "ref_text": ref["ref_text"]})
                idx += 1

    for name, q in (("probe_named", named), ("probe_cloned", cloned)):
        path = args.out_dir / f"{name}_queue.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e in q:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{len(q):5d} items -> {path}")


if __name__ == "__main__":
    main()
