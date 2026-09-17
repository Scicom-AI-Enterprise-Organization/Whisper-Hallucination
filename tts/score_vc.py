#!/usr/bin/env python3
"""Score the VC / cloning candidates: did the words survive, and did the voice actually move?

Three numbers per clip, because any one of them alone picks the wrong winner:

  cer         ASR round-trip against the phrase, language forced -- same judge as
              `score_tts.py`, for the same reason: a positive example is useless if Whisper
              cannot recover the phrase from it.
  d_cer       cer MINUS the source clip's own cer. The source is TTS with its own errors;
              what a converter is responsible for is the DEGRADATION it adds.
  sim_tgt     cosine between the output's x-vector and the target speaker's reference.
  sim_src     cosine to the SOURCE clip's speaker. This is the trap: a converter that
              returns its input unchanged scores a perfect 0.0 d_cer and is worthless.
              A real conversion moves sim_tgt up and sim_src down together.

The headline ranking is by `d_cer` among systems that actually converted
(`sim_tgt > sim_src`); anything failing that gate is reported but not ranked.

    .venv_bench/bin/python tts/score_vc.py --systems tts/vc_out/* --device cuda:6
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer  # noqa: E402

SV_MODEL = "microsoft/wavlm-base-plus-sv"
SR = 16000


def degenerate(phrase: str) -> str:
    """Whisper loop artefacts that got captured as lexicon 'phrases' (CLAUDE.md).

    `त र`, `સ સ સ સ સ સ સ` -- no TTS renders them as words and no ASR recovers them, so they
    score CER ~20 for every system and swamp the per-language mean for whichever language
    they are filed under. Three of the 51 phrases; excluded from the aggregates and counted
    separately rather than silently dropped.
    """
    toks = phrase.split()
    if len(toks) > 1 and len(set(toks)) == 1:
        return "single repeated token"
    if len(set(phrase.replace(" ", ""))) <= 2:
        return "<=2 distinct characters"
    return ""


def read16k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


class Judges:
    """Whisper (words) + WavLM-sv (voice), each cached by path -- sources repeat per target."""

    def __init__(self, asr_model, device):
        from transformers import (AutoFeatureExtractor, WavLMForXVector,
                                  WhisperForConditionalGeneration, WhisperProcessor)
        self.device = device
        self.proc = WhisperProcessor.from_pretrained(asr_model)
        self.asr = WhisperForConditionalGeneration.from_pretrained(
            asr_model, dtype=torch.float16).to(device).eval()
        self.feat = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.sv = WavLMForXVector.from_pretrained(SV_MODEL).to(device).eval()
        self._hyp, self._emb = {}, {}

    def transcribe(self, path, lang):
        key = (str(path), lang)
        if key not in self._hyp:
            x = read16k(path)
            f = self.proc(x, sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                gen = self.asr.generate(f.input_features.to(self.device, torch.float16),
                                        language=lang if len(lang) == 2 else None,
                                        task="transcribe", max_new_tokens=160, num_beams=1)
            self._hyp[key] = self.proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return self._hyp[key]

    def xvector(self, path):
        key = str(path)
        if key not in self._emb:
            x = read16k(path)
            inp = self.feat([x], sampling_rate=SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                e = self.sv(**{k: v.to(self.device) for k, v in inp.items()}).embeddings
            self._emb[key] = torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()
        return self._emb[key]


def score_system(sysdir: Path, targets: dict, j: Judges) -> dict:
    rows = list(csv.DictReader((sysdir / "manifest.csv").open(encoding="utf-8")))

    # Score every system on the same set of (target, phrase) pairs. Systems differ in
    # whether they rendered the two duplicated phrases once or twice, and letting the
    # duplicates through would weight two easy English phrases double for some systems only.
    seen, deduped = set(), []
    for r in rows:
        k = (r["target"], r["phrase"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    rows = deduped

    per_lang, details = defaultdict(list), []
    agg = defaultdict(list)

    n_degenerate = 0
    for r in rows:
        lang, phrase = r["lang"], r["phrase"]
        if degenerate(phrase):
            n_degenerate += 1
            continue
        if r.get("status") != "ok" or not r["audio_filepath"]:
            per_lang[lang].append(1.0)
            details.append({**r, "hyp": "", "cer": 1.0, "src_cer": None,
                            "d_cer": None, "sim_tgt": None, "sim_src": None})
            continue

        out_path = sysdir / r["audio_filepath"]
        hyp = j.transcribe(out_path, lang)
        c = cer(phrase, hyp)
        src_c = cer(phrase, j.transcribe(r["src_audio"], lang)) if Path(r["src_audio"]).exists() else None

        # Duration ratio against the source: a cloning model that keeps talking past a
        # two-word phrase produces a clip that is not a clean positive, however its CER
        # reads. Conversion systems are pinned near 1.0 by construction; cloning ones
        # are not, and Scicom's cloning path runs to the token cap on short targets.
        dur_ratio = None
        if Path(r["src_audio"]).exists():
            try:
                src_dur = len(read16k(r["src_audio"])) / SR
                out_dur = float(r.get("duration_s") or 0) or len(read16k(out_path)) / SR
                if src_dur > 0:
                    dur_ratio = out_dur / src_dur
            except Exception:
                dur_ratio = None

        e_out = j.xvector(out_path)
        ref = targets.get(r["target"], {}).get("ref_path")
        sim_tgt = float(e_out @ j.xvector(ref)) if ref and Path(ref).exists() else None
        sim_src = float(e_out @ j.xvector(r["src_audio"])) if Path(r["src_audio"]).exists() else None

        per_lang[lang].append(c)
        for k, v in (("cer", c), ("sim_tgt", sim_tgt), ("sim_src", sim_src),
                     ("dur_ratio", dur_ratio),
                     ("d_cer", None if src_c is None else c - src_c)):
            if v is not None:
                agg[k].append(v)
        details.append({**r, "hyp": hyp, "cer": round(c, 4),
                        "src_cer": None if src_c is None else round(src_c, 4),
                        "d_cer": None if src_c is None else round(c - src_c, 4),
                        "sim_tgt": None if sim_tgt is None else round(sim_tgt, 4),
                        "sim_src": None if sim_src is None else round(sim_src, 4),
                        "dur_ratio": None if dur_ratio is None else round(dur_ratio, 3)})

    # A system with a fixed speaker inventory (Scicom's Multilingual-Expressive) has no
    # reference clip to aim at, so "did it reach the target" is undefined rather than failed.
    targeted = any(targets.get(r["target"], {}).get("ref_path") for r in rows)

    mean = lambda k: round(float(np.mean(agg[k])), 4) if agg[k] else None
    scored = [r for r in rows if not degenerate(r["phrase"])]
    ok = sum(r.get("status") == "ok" for r in scored)
    return {
        "targeted": targeted,
        "n": len(scored), "n_degenerate_excluded": n_degenerate, "synth_ok": ok,
        "ok_rate": round(ok / len(scored), 4) if scored else 0.0,
        "mean_cer": mean("cer"), "mean_d_cer": mean("d_cer"),
        "median_d_cer": round(float(np.median(agg["d_cer"])), 4) if agg["d_cer"] else None,
        "mean_sim_tgt": mean("sim_tgt"), "mean_sim_src": mean("sim_src"),
        "median_dur_ratio": round(float(np.median(agg["dur_ratio"])), 3) if agg["dur_ratio"] else None,
        "overlong_rate": (round(float(np.mean([r > 2.0 for r in agg["dur_ratio"]])), 4)
                          if agg["dur_ratio"] else None),
        "converted": bool(targeted and agg["sim_tgt"] and agg["sim_src"]
                          and np.mean(agg["sim_tgt"]) > np.mean(agg["sim_src"])),
        "langs": len([l for l, v in per_lang.items() if any(c < 1.0 for c in v)]),
        "per_lang_cer": {k: round(float(np.mean(v)), 4) for k, v in sorted(per_lang.items())},
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", type=Path, required=True)
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_scores.json")
    args = ap.parse_args()

    targets = {t["id"]: t for t in json.loads((args.targets / "targets.json").read_text())}
    for t in targets.values():
        t["ref_path"] = str((args.targets / t["id"] / "ref.flac").resolve())

    j = Judges(args.model, args.device)
    results = {}
    for d in args.systems:
        if not (d / "manifest.csv").exists():
            print(f"[skip] {d} has no manifest")
            continue
        print(f"[score] {d.name}", flush=True)
        results[d.name] = score_system(d, targets, j)

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    hdr = (f"{'system':<22}{'ok':>7}{'langs':>6}{'CER':>8}{'dCER':>8}{'sim_tgt':>9}{'sim_src':>9}"
           f"{'dur×':>7}{'>2×':>6}  verdict")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in sorted(results.items(), key=lambda kv: (not kv[1]["converted"],
                                                          kv[1]["mean_d_cer"] if kv[1]["mean_d_cer"] is not None else 9)):
        f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
        if not r["targeted"]:
            verdict = "untargeted (fixed speaker inventory -- cannot aim at a reference)"
        elif r["converted"]:
            verdict = "converted"
        else:
            verdict = "NO CONVERSION (sim_src >= sim_tgt)"
        ov = '-' if r.get('overlong_rate') is None else f"{r['overlong_rate']:.0%}"
        print(f"{name:<22}{r['ok_rate']:>7.2f}{r['langs']:>6}{f(r['mean_cer']):>8}{f(r['mean_d_cer']):>8}"
              f"{f(r['mean_sim_tgt']):>9}{f(r['mean_sim_src']):>9}"
              f"{f(r.get('median_dur_ratio'), 2):>7}{ov:>6}  {verdict}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
