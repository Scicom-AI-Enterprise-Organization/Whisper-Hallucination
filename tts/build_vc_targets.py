#!/usr/bin/env python3
"""Build the fixed target-speaker pool every VC / voice-cloning candidate converts TO.

Four targets, two Malay and two English, because the question is cross-lingual timbre
transfer: the sources are lexicon phrases in 23 languages, and a converter that only holds
up when source and target share a language is no use for the positive pool.

Speaker identity comes from EMBEDDINGS, not metadata. The Emilia clips are named by YouTube
video, and a video is not a speaker -- most of them are interviews with two or more voices.
So each candidate pool is embedded with WavLM-sv, and only the clips within `--max-cos` of
the medoid are kept. LibriSpeech has no speaker id in our published config either, so the
same medoid pass recovers its speakers from the audio.

Each target gets both shapes of reference the candidates need:
  ref.flac   one ~10 s clip   -- seed-vc, OpenVoice, CosyVoice (one-shot)
  pool/      several minutes  -- knn-vc, which matches against a whole speaker's features

    python tts/build_vc_targets.py --device cuda:6
"""
import argparse, io, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
SV_MODEL = "microsoft/wavlm-base-plus-sv"
VIDEO_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]_\d+\.flac$")


def embed(paths_or_arrays, feat, model, device, batch=8):
    """x-vector per clip, L2-normalised so a dot product is the cosine."""
    out = []
    for i in range(0, len(paths_or_arrays), batch):
        chunk = paths_or_arrays[i:i + batch]
        waves = [x if isinstance(x, np.ndarray) else sf.read(x, dtype="float32")[0] for x in chunk]
        waves = [w.mean(1) if w.ndim > 1 else w for w in waves]
        inp = feat(waves, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = model(**{k: v.to(device) for k, v in inp.items()}).embeddings
        e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        out.append(e)
    return np.concatenate(out) if out else np.zeros((0, 512), "float32")


def medoid_keep(embs, max_cos):
    """Index of the medoid, and the members within `max_cos` cosine of it."""
    sim = embs @ embs.T
    med = int(sim.mean(1).argmax())
    keep = np.nonzero(sim[med] >= max_cos)[0]
    return med, keep


def emilia_texts():
    """audio_filepath -> transcript, from the training corpus manifest."""
    man = ROOT / "corpus" / "train.jsonl"
    if not man.exists():
        return {}
    out = {}
    for line in man.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") == "emilia":
            out[r["audio_filepath"]] = r.get("text", "")
    return out


def load_emilia_pools(n_videos):
    """Candidate Malay pools, keyed by YouTube video id -- biggest first."""
    d = ROOT / "audio_train" / "emilia" / "flac"
    if not d.exists():
        return {}
    by_vid = defaultdict(list)
    for p in sorted(d.glob("*.flac")):
        m = VIDEO_RE.search(p.name)
        if m:
            by_vid[m.group(1)].append(p)
    ranked = sorted(by_vid.items(), key=lambda kv: -len(kv[1]))
    return dict(ranked[:n_videos])


def load_librispeech(limit):
    """English candidates from our own published benchmark -- already cached on the box."""
    from datasets import Audio, load_dataset
    ds = load_dataset("Scicom-intl/Whisper-Hallucination", "librispeech_test_clean",
                      split="test").cast_column("audio", Audio(decode=False))
    ds = ds.select(range(min(limit, len(ds))))
    waves, texts = [], []
    for r in ds:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        waves.append(x.mean(1) if x.ndim > 1 else x)
        texts.append(r.get("text", ""))
    return waves, texts


def write_target(out, tid, lang, source, clips, texts, ref_seconds, pool_seconds, meta,
                 ref_short_seconds=6.0):
    """clips: float32 arrays at SR, best-first; texts: their transcripts, same order.

    The transcript matters because zero-shot cloning (CosyVoice) is prompted with reference
    audio AND what is said in it; a wrong prompt text degrades the clone.
    """
    tdir = out / tid
    (tdir / "pool").mkdir(parents=True, exist_ok=True)
    for f in (tdir / "pool").glob("*.flac"):
        f.unlink()

    # One-shot reference: concatenate from the best clips until ref_seconds, so a target
    # built from 2 s utterances still gets a reference long enough to characterise a voice.
    # A second, SHORT reference: zero-shot cloning degrades when the prompt is much longer
    # than the text being synthesised, and these phrases are two or three words. Conversion
    # has the opposite preference, so both shapes get written and each system takes its own.
    sf.write(tdir / "ref_short.flac", clips[0][:int(ref_short_seconds * SR)], SR)

    ref, ref_text, total = [], [], 0.0
    for c, t in zip(clips, texts):
        ref.append(c)
        ref.append(np.zeros(int(0.15 * SR), "float32"))
        ref_text.append((t or "").strip())
        total += len(c) / SR + 0.15
        if total >= ref_seconds:
            break
    sf.write(tdir / "ref.flac", np.concatenate(ref), SR)

    kept, secs = 0, 0.0
    for i, c in enumerate(clips):
        if secs >= pool_seconds:
            break
        sf.write(tdir / "pool" / f"{i:04d}.flac", c, SR)
        kept, secs = kept + 1, secs + len(c) / SR
    short_s = min(ref_short_seconds, len(clips[0]) / SR)
    return {"id": tid, "lang": lang, "source": source, "ref_s": round(total, 2),
            "ref_text": " ".join(x for x in ref_text if x),
            "ref_short_s": round(short_s, 2),
            # Truncating the audio truncates the sentence, and a prompt text that overruns
            # its audio hurts as much as one that is too long -- so keep only as many words
            # as the clip plausibly covers.
            "ref_short_text": " ".join((texts[0] or "").split()[:max(1, int(short_s * 2.5))]),
            "pool_clips": kept, "pool_s": round(secs, 1), **meta}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--ref-seconds", type=float, default=10.0)
    ap.add_argument("--pool-seconds", type=float, default=240.0)
    ap.add_argument("--max-cos", type=float, default=0.90, help="cosine to the medoid to count as the same voice")
    ap.add_argument("--n-malay", type=int, default=2)
    ap.add_argument("--n-english", type=int, default=2)
    ap.add_argument("--librispeech-limit", type=int, default=800)
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, WavLMForXVector
    feat = AutoFeatureExtractor.from_pretrained(SV_MODEL)
    sv = WavLMForXVector.from_pretrained(SV_MODEL).to(args.device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    targets = []

    # --- Malay: one target per video, keeping only the dominant voice in it ----------
    etexts = emilia_texts()
    pools = load_emilia_pools(n_videos=args.n_malay * 3)
    for vid, paths in pools.items():
        if len(targets) >= args.n_malay:
            break
        paths = paths[:120]
        embs = embed(paths, feat, sv, args.device)
        med, keep = medoid_keep(embs, args.max_cos)
        if len(keep) < 20:                       # too fragmented to call one speaker
            print(f"[skip] {vid}: only {len(keep)}/{len(paths)} clips near the medoid")
            continue
        order = sorted(keep, key=lambda i: -float(embs[med] @ embs[i]))
        clips, clip_texts = [], []
        for i in order:
            x, _ = sf.read(paths[i], dtype="float32")
            clips.append(x.mean(1) if x.ndim > 1 else x)
            clip_texts.append(etexts.get(str(paths[i]), ""))
        tid = f"ms_{vid}"
        targets.append(write_target(args.out, tid, "ms", f"Malaysian-Emilia [{vid}]", clips,
                                    clip_texts, args.ref_seconds, args.pool_seconds,
                                    {"n_candidates": len(paths), "n_same_voice": len(keep),
                                     "mean_cos": round(float((embs[keep] @ embs[med]).mean()), 4)}))
        print(f"[ok  ] {tid}: {len(keep)}/{len(paths)} clips are one voice")

    # --- English: recover speakers from LibriSpeech by greedy medoid peeling ---------
    waves, wtexts = load_librispeech(args.librispeech_limit)
    if waves:
        embs = embed(waves, feat, sv, args.device)
        alive = np.ones(len(waves), bool)
        for _ in range(args.n_english):
            idx = np.nonzero(alive)[0]
            if len(idx) < 20:
                break
            sub = embs[idx]
            med, keep = medoid_keep(sub, args.max_cos)
            if len(keep) < 10:
                alive[idx[med]] = False
                continue
            order = sorted(keep, key=lambda i: -float(sub[med] @ sub[i]))
            clips = [waves[idx[i]] for i in order]
            clip_texts = [wtexts[idx[i]] for i in order]
            tid = f"en_{idx[med]:04d}"
            targets.append(write_target(args.out, tid, "en", "LibriSpeech test-clean", clips,
                                        clip_texts, args.ref_seconds, args.pool_seconds,
                                        {"n_candidates": int(alive.sum()), "n_same_voice": len(keep)}))
            print(f"[ok  ] {tid}: {len(keep)} clips are one voice")
            alive[idx[keep]] = False

    (args.out / "targets.json").write_text(json.dumps(targets, indent=2, ensure_ascii=False))
    print(f"\n{len(targets)} targets -> {args.out}")
    for t in targets:
        print(f"  {t['id']:<16} {t['lang']}  ref {t['ref_s']:>5.1f}s  pool {t['pool_s']:>6.1f}s "
              f"({t['pool_clips']} clips)  {t['source']}")


if __name__ == "__main__":
    main()
