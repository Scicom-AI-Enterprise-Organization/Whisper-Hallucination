#!/usr/bin/env python3
"""Mine wild audio for clips that actually make Whisper hallucinate or loop.

The benchmark's negative arms are built stimuli: synthesised silence, FMA excerpts, FSD50K
noise, generated reduplication. They are exact but they are not what production audio looks
like. This finds the real thing in public corpora, and it does so WITHOUT human labelling,
because two of the three failure signatures are self-evident:

  loop            a token run >= `--loop-threshold`. A correct transcript almost never
                  repeats one token six times; no reference needed.
  blank_speech    Silero VAD finds NO speech in the clip, yet the model emitted words. Any
                  output over voice-free audio is invented; no reference needed.
  lexicon_hit     the transcript is exactly a known hallucination phrase AND the VAD found no
                  speech. On its own this reason is useless -- see below.
  insertion       the corpus ships a reference and the model emitted much MORE than it, with
                  the surplus containing a lexicon phrase. This is the podcast failure: not
                  silence filled with text, but real speech over a music bed or crosstalk that
                  the model pads with `thanks for watching`. GigaSpeech and People's Speech
                  both carry transcripts, so the signal is free there; AMI-style blank clips
                  never fire it.

Nothing here calls a clip a hallucination because the transcript disagrees with a reference;
that just finds ordinary ASR errors.

**An RMS threshold does not work and the first run proved it.** Energy below -45 dBFS was used
as the blank test, which flagged 2,920 of 6,000 AMI clips: AMI's headset mics simply record
quietly (-45 to -51 dBFS) while containing perfectly good speech, so "because yeah, I mean,
the wave data are obviously not going..." was filed as a hallucination. Corpus level is not
evidence of silence. Silero VAD replaces it, and `lexicon_hit` no longer fires on its own --
otherwise every backchannel "Yeah." in a meeting counts, which is another 1,404 false hits.

Sources are streamed, so nothing is downloaded in full:

    python scripts/mine_wild_hallucinations.py --source yodas --lang ms --max-clips 5000
    python scripts/mine_wild_hallucinations.py --source peoples_speech --max-clips 5000
    python scripts/mine_wild_hallucinations.py --source malaysian_podcast --max-clips 5000
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import max_ngram_repeat, normalise  # noqa: E402

SR = 16000

# Streamed, permissively licensed, and wild: real recordings nobody made for a benchmark.
# Verified to stream with `Audio(decode=False)` on 2026-09-23. Three obvious candidates do
# NOT work and are left out rather than silently failing mid-run:
#   espnet/yodas2                          script-based loader, unsupported by datasets>=3
#   malaysia-ai/malaysian-podcast-youtube  split ZIP64, the Malaysian-Emilia problem again
#   malaysia-ai/malaysian-youtube          demands torchcodec even with decode=False
# Yield is a function of how rough the audio is. Measured per 8,000 clips heard: AMI
# meetings 235, Earnings-22 31, VoxPopuli 4, People's Speech CLEAN 1. Curated read speech
# recorded for a dataset barely fails; podcasts, YouTube and noisy crowd-sourced audio are
# where hallucinations actually happen, so those are the sources that matter here.
SOURCES = {
    "gigaspeech":     dict(repo="speechcolab/gigaspeech", config="l", split="train",
                           note="podcasts + YouTube, the noisy real-world case"),
    "gigaspeech_xs":  dict(repo="speechcolab/gigaspeech", config="xs", split="train",
                           note="same, small config for a quick pass"),
    "peoples_dirty":  dict(repo="MLCommons/peoples_speech", config="dirty", split="train",
                           note="the NOISY subset -- `clean` yields ~0.01%"),
    "peoples_dirty_sa": dict(repo="MLCommons/peoples_speech", config="dirty_sa", split="train",
                             note="noisy, self-attributed licence subset"),
    "audioset":       dict(repo="agkphysics/AudioSet", config=None, split="train",
                           note="YouTube audio, mostly environmental -- the classic trigger"),
    "peoples_speech": dict(repo="MLCommons/peoples_speech", config="clean", split="train",
                           note="30k h CC-BY / CC-BY-SA, read + spontaneous"),
    "ami":            dict(repo="edinburghcstr/ami", config="ihm", split="train",
                           note="real meetings, spontaneous, overlapping speech"),
    "voxpopuli":      dict(repo="facebook/voxpopuli", config="en", split="train",
                           note="European Parliament, CC0, long pauses"),
    "earnings22":     dict(repo="distil-whisper/earnings22", config="chunked", split="test",
                           note="earnings calls -- the corpus HALAS annotated"),
}


_VAD = None


def speech_report(x: np.ndarray, sr: int = SR) -> dict:
    """Silero VAD plus the clip's level, so a decision never rests on loudness alone."""
    global _VAD
    if _VAD is None:
        from silero_vad import load_silero_vad
        # ONNX, not the torch path: silero-vad pulls a torchaudio that collides with the
        # benchmark venv's torch ("cannot import name 'ScalingType'"). The ONNX runtime has
        # no such dependency and gives the same model.
        _VAD = load_silero_vad(onnx=True)
    from silero_vad import get_speech_timestamps
    rms_db = round(float(20 * np.log10(max(np.sqrt((x ** 2).mean()), 1e-12))), 2)
    # Normalise before the VAD: it is trained on speech at a sane level, and a quietly
    # recorded corpus should not read as silence. Peak-normalising keeps quiet REAL speech
    # detectable while leaving true room tone below the model's threshold.
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    xn = (x / peak).astype("float32") if peak > 1e-6 else x.astype("float32")
    try:
        ts = get_speech_timestamps(xn, _VAD, sampling_rate=sr)
    except Exception:
        return {"rms_db": rms_db, "speech_s": -1.0, "speech_frac": -1.0}
    voiced = sum(t["end"] - t["start"] for t in ts) / sr
    return {"rms_db": rms_db, "speech_s": round(voiced, 3),
            "speech_frac": round(voiced / max(len(x) / sr, 1e-9), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--lang", default=None, help="config/language for sources that need one")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--out", type=Path, default=Path("audio_wild"))
    ap.add_argument("--max-clips", type=int, default=2000, help="clips to LISTEN to")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="skip longer clips")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--loop-threshold", type=int, default=6)
    ap.add_argument("--min-blank-dur", type=float, default=0.4,
                    help="Silero needs roughly 250 ms of audio to judge; below that a real "
                         "backchannel reads as speech_frac 0.0. A 0.2 s AMI clip at -28 dBFS "
                         "is a starved VAD, not silence, so short clips cannot claim `blank`")
    ap.add_argument("--min-insertion-ratio", type=float, default=1.6,
                    help="hypothesis/reference character ratio above which the surplus is "
                         "checked for a lexicon phrase")
    ap.add_argument("--max-speech-frac", type=float, default=0.02,
                    help="a clip with less than this fraction of VAD speech counts as "
                         "voice-free. Not a loudness threshold -- see the docstring")
    ap.add_argument("--lexicon", type=Path, default=ROOT / "lexicon" / "combined_lexicon.csv")
    args = ap.parse_args()

    lex = set()
    if args.lexicon.exists():
        with args.lexicon.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                p = normalise(r.get("phrase", ""))
                if p:
                    lex.add(p)
    print(f"[wild] lexicon: {len(lex)} phrases", flush=True)

    import soundfile as sf
    from datasets import Audio, load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    spec = SOURCES[args.source]
    cfg = args.lang or spec["config"]
    print(f"[wild] streaming {spec['repo']} ({spec['note']}) config={cfg}", flush=True)
    ds = load_dataset(spec["repo"], cfg, split=spec["split"], streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    proc = WhisperProcessor.from_pretrained(args.model)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()

    out_dir = args.out / args.source / (cfg or "default")
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    rows, batch, seen, kept, t0 = [], [], 0, 0, time.time()

    def flush(batch):
        nonlocal kept
        if not batch:
            return
        feats = proc([b["x"] for b in batch], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            gen = asr.generate(feats.input_features.to(args.device, torch.float16),
                               task="transcribe", max_new_tokens=220, num_beams=1)
        for b, hyp in zip(batch, proc.batch_decode(gen, skip_special_tokens=True)):
            hyp = hyp.strip()
            norm = normalise(hyp)
            run = max_ngram_repeat(hyp, 1)
            vad = b["vad"]
            blank = (0.0 <= vad["speech_frac"] <= args.max_speech_frac
                     and b["dur"] >= args.min_blank_dur)
            reasons = []
            if run >= args.loop_threshold:
                reasons.append("loop")
            if norm and blank:
                reasons.append("blank_speech")
            # Only meaningful together with blank: a meeting is full of genuine "Yeah."
            if norm and blank and norm in lex:
                reasons.append("lexicon_hit")
            # Insertion: real speech, but the model added text that is not in the reference
            # and the addition is a known hallucination phrase.
            ref = normalise(b.get("ref") or "")
            if ref and norm and len(norm) > args.min_insertion_ratio * len(ref):
                extra = norm.replace(ref, " ").strip()
                if extra and any(pz in extra for pz in lex if len(pz) >= 8):
                    reasons.append("insertion")
            if not reasons:
                continue
            rel = f"wav/{args.source}_{b['idx']:07d}.flac"
            sf.write(out_dir / rel, b["x"], SR, format="FLAC", subtype="PCM_16")
            rows.append({"id": f"{args.source}_{b['idx']:07d}", "audio_filepath": rel,
                         "source": spec["repo"], "config": cfg or "", "hyp": hyp[:300],
                         "reasons": "|".join(reasons), "max_token_run": run,
                         "duration_s": round(b["dur"], 3),
                         "reference_text": (b.get("ref") or "")[:300], **vad,
                         "source_id": str(b["src_id"])[:120]})
            kept += 1

    for i, row in enumerate(ds):
        if seen >= args.max_clips:
            break
        a = row.get("audio") or {}
        try:
            x, sr = sf.read(__import__("io").BytesIO(a["bytes"]), dtype="float32")
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        dur = len(x) / SR
        if dur > args.max_seconds or dur < 0.2:
            continue
        seen += 1
        batch.append({"x": x, "dur": dur, "idx": i, "vad": speech_report(x),
                      "ref": row.get("text") or row.get("transcription")
                             or row.get("raw_text") or row.get("sentence") or "",
                      "src_id": row.get("id") or a.get("path") or i})
        if len(batch) >= args.batch_size:
            flush(batch); batch = []
            if seen % (args.batch_size * 20) == 0:
                el = time.time() - t0
                print(f"  heard {seen}  kept {kept}  ({seen/max(el,1e-9):.1f} clip/s)", flush=True)
    flush(batch)

    man = out_dir / "manifest.csv"
    if rows:
        with man.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    summary = {"source": spec["repo"], "config": cfg, "heard": seen, "kept": kept,
               "model": args.model,
               "by_reason": {r: sum(r in x["reasons"].split("|") for x in rows)
                             for r in ("loop", "blank_speech", "lexicon_hit", "insertion")}}
    (out_dir / "mine_report.json").write_text(json.dumps(summary, indent=2))
    print(f"[wild] heard {seen}, kept {kept} -> {man}")
    print(json.dumps(summary["by_reason"], indent=1))


if __name__ == "__main__":
    main()
