#!/usr/bin/env python3
"""Decide, per language, which engine and which voice to synthesise lexicon phrases with.

The choice is not one model. Two things fall out of the measurements:

  * Speaker choice matters about as much as model choice — across the four names measured,
    mean CER runs from 0.273 (Serena) to 0.504 (Grace, the name the model card recommends).
  * Neither engine wins everywhere. Multilingual-Expressive with its best speaker beats
    OmniVoice in most languages, loses in a few, and ties where both already score 0.

So this writes a per-language plan: the engine and voice with the lowest measured CER, with
OmniVoice as the fallback wherever it wins or the language was never measured.

    python tts/build_voice_plan.py --scores tts/vc_scores.json --out tts/lexicon_voice_plan.json
"""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 0.02          # ignore differences smaller than this; they are noise at n≈2/lang


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=ROOT / "tts" / "vc_scores.json")
    ap.add_argument("--system", default="scicom_untargeted", help="the speaker-name-mode run")
    ap.add_argument("--speakers", type=Path, default=ROOT / "tts" / "expressive_speakers.json",
                    help="verified names; anything else is dropped rather than trusted")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "lexicon_voice_plan.json")
    args = ap.parse_args()

    # An unknown speaker name does not raise in Multilingual-Expressive -- it conditions on a
    # token the fine-tune never saw and silently returns unconditioned audio, which reads as a
    # bad language rather than a bad name. `multilingual-tts_audio_Rahman` got into an earlier
    # plan that way and took pl/ta/ur down with it. So the plan may only name verified speakers.
    valid = set()
    if args.speakers.exists():
        spk = json.loads(args.speakers.read_text())
        valid = set(spk.get("all_speakers", {})) or set(spk.get("multilingual_tts", {}))

    d = json.loads(args.scores.read_text())
    rows = [r for r in d[args.system]["details"] if r.get("status") == "ok"]

    by_lang_spk, src_by_lang = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    dropped = set()
    for r in rows:
        if valid and r["target"] not in valid:
            dropped.add(r["target"]); continue
        by_lang_spk[r["lang"]][r["target"]].append(r["cer"])
        if r.get("src_cer") is not None:
            src_by_lang[r["lang"]].append(r["src_cer"])

    if dropped:
        print(f"[drop] speakers not in {args.speakers.name}: {sorted(dropped)}")

    plan, wins = {}, defaultdict(int)
    for lang in sorted(by_lang_spk):
        per_spk = {s: statistics.mean(v) for s, v in by_lang_spk[lang].items()}
        best_spk = min(per_spk, key=per_spk.get)
        scicom, omni = per_spk[best_spk], statistics.mean(src_by_lang.get(lang, [1.0]))

        if scicom <= omni - MARGIN:
            engine, voice, cer, why = "multilingual-expressive", best_spk, scicom, "lower CER"
        elif omni <= scicom - MARGIN:
            engine, voice, cer, why = "omnivoice", None, omni, "lower CER"
        else:
            # A tie is usually both at 0.000; prefer the permissive, enumerable-coverage one.
            engine, voice, cer, why = "omnivoice", None, omni, "tie — Apache-2.0 preferred"
        wins[engine] += 1
        plan[lang] = {"engine": engine, "voice": voice, "measured_cer": round(cer, 4),
                      "scicom_best": round(scicom, 4), "scicom_best_voice": best_spk,
                      "omnivoice": round(omni, 4), "reason": why}

    speaker_means = {s: round(statistics.mean([c for l in by_lang_spk.values()
                                               for c in l.get(s, [])]), 4)
                     for s in {s for l in by_lang_spk.values() for s in l}}
    out = {
        "verified_speakers_only": bool(valid),
        "dropped_unverified": sorted(dropped),
        "default_for_unmeasured_languages": {
            "engine": "omnivoice",
            "why": "97/100 languages enumerable and Apache-2.0; Multilingual-Expressive's "
                   "coverage is untagged, so it is only chosen where it was measured to win",
        },
        "speaker_mean_cer": dict(sorted(speaker_means.items(), key=lambda kv: kv[1])),
        "per_language": plan,
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    print(f"{'lang':<6}{'engine':<26}{'voice':<34}{'CER':>7}")
    print("-" * 73)
    for lang, p in plan.items():
        print(f"{lang:<6}{p['engine']:<26}{(p['voice'] or '—'):<34}{p['measured_cer']:>7.3f}")
    print(f"\nengine split: {dict(wins)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
