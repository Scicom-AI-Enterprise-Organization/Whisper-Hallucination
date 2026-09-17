#!/usr/bin/env python3
"""Shared plumbing for the VC / voice-cloning ablation: sources, targets, output manifests.

Every candidate writes the same manifest so `score_vc.py` can score them side by side --
one row per (phrase, target), with the source clip it came from recorded, because the
interesting number is DEGRADATION against that source, not absolute CER.
"""
import csv, json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 16000

FIELDS = ["lang", "phrase", "halluc_count", "target", "target_lang", "source_system",
          "src_audio", "audio_filepath", "duration_s", "status", "error"]


def patch_torchaudio_load():
    """Route `torchaudio.load` through soundfile.

    torchaudio >= 2.9 delegates load() to torchcodec and raises if it is missing -- the same
    torchcodec tax `datasets >= 5` charges (CLAUDE.md). Nothing here needs a decoder beyond
    FLAC/WAV, so soundfile covers it and torchcodec stays uninstalled.
    """
    import torch, torchaudio

    def _load(path, normalize=True, **kw):
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(x.T)), sr

    torchaudio.load = _load


def load_sources(sysdir: Path) -> list[dict]:
    """Rows of a TTS output dir, with `abs_path` resolved and failures dropped."""
    rows = list(csv.DictReader((sysdir / "manifest.csv").open(encoding="utf-8")))
    out = []
    for r in rows:
        if r.get("status") != "ok" or not r.get("audio_filepath"):
            continue
        r["abs_path"] = str((sysdir / r["audio_filepath"]).resolve())
        out.append(r)
    return out


def load_targets(tdir: Path) -> list[dict]:
    """Target speakers with ref clip and pool paths resolved."""
    targets = json.loads((tdir / "targets.json").read_text())
    for t in targets:
        t["ref_path"] = str((tdir / t["id"] / "ref.flac").resolve())
        t["ref_short_path"] = str((tdir / t["id"] / "ref_short.flac").resolve())
        t["pool_paths"] = [str(p) for p in sorted((tdir / t["id"] / "pool").glob("*.flac"))]
    return targets


def pairs(sources: list[dict], targets: list[dict], skip_same_lang: bool = False):
    """Every (source clip, target speaker) pair -- the full grid, deterministically ordered."""
    for t in targets:
        for s in sources:
            if skip_same_lang and s["lang"] == t["lang"]:
                continue
            yield s, t


class Writer:
    """Incremental manifest writer, resumable across process death.

    CosyVoice's ONNX speech tokenizer can take the process down with SIGFPE, which no
    `except` can catch -- so the manifest is append-only and `--resume` picks up where the
    corpse left off. The pair being attempted is recorded first: if the next run finds that
    same pair still unfinished, it is poison, and gets written off as a failure rather than
    crashing the run again forever.
    """

    def __init__(self, outdir: Path, system: str, source_system: str, resume: bool = False):
        self.dir = outdir / system
        (self.dir / "wav").mkdir(parents=True, exist_ok=True)
        self.source_system = source_system
        self.attempt_file = self.dir / "_attempt.json"
        self.crash_file = self.dir / "_crashes.json"
        manifest = self.dir / "manifest.csv"

        self.done, self.poison, self.crashes = set(), set(), {}
        if resume and manifest.exists() and manifest.stat().st_size:
            with manifest.open(encoding="utf-8") as fh:
                # Must match key() exactly -- keyed on the source clip, not its text.
                self.done = {(r["target"], r["src_audio"]) for r in csv.DictReader(fh)}
            if self.crash_file.exists():
                self.crashes = json.loads(self.crash_file.read_text())
            if self.attempt_file.exists():
                last = tuple(json.loads(self.attempt_file.read_text()))
                if last not in self.done:
                    # One crash may be luck of the scheduler rather than the input, and
                    # writing a good pair off as poison would understate the system. Two
                    # crashes on the same pair is the input.
                    k = "\x1f".join(last)
                    self.crashes[k] = self.crashes.get(k, 0) + 1
                    self.crash_file.write_text(json.dumps(self.crashes))
                    if self.crashes[k] >= 2:
                        self.poison.add(last)
            self.fh = manifest.open("a", newline="", encoding="utf-8")
            self.w = csv.DictWriter(self.fh, fieldnames=FIELDS)
        else:
            self.fh = manifest.open("w", newline="", encoding="utf-8")
            self.w = csv.DictWriter(self.fh, fieldnames=FIELDS)
            self.w.writeheader()
            self.fh.flush()                        # a signal must not take the header with it
        self.n_ok = self.n_fail = 0

    def key(self, src: dict, tgt: dict):
        """Identity of a (source clip, target) pair.

        Keyed on the clip path, NOT the phrase: `phrases.jsonl` holds 51 entries with only
        49 distinct texts, so a phrase-keyed set silently skipped the repeats and a system
        that used `skip()` ended up scored on fewer clips than one that did not.
        """
        return (tgt["id"], src["abs_path"])

    def skip(self, src: dict, tgt: dict) -> bool:
        return self.key(src, tgt) in self.done

    def attempting(self, src: dict, tgt: dict):
        """Record the pair about to be tried, so a hard crash is attributable to it."""
        self.attempt_file.write_text(json.dumps(self.key(src, tgt)))

    def add(self, src: dict, tgt: dict, wave, sr: int, idx: int, error: str = ""):
        rel, dur = "", 0.0
        if wave is not None and len(wave):
            wave = np.asarray(wave, dtype="float32").squeeze()
            if wave.ndim > 1:
                wave = wave.mean(0) if wave.shape[0] < wave.shape[1] else wave.mean(1)
            if sr != SR:                       # one rate for everything the scorer touches
                import librosa
                wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)
            peak = float(np.abs(wave).max() or 0.0)
            if peak > 1.0:
                wave = wave / peak
            rel = f"wav/{tgt['id']}_{idx:04d}.flac"
            sf.write(self.dir / rel, wave, SR)
            dur = round(len(wave) / SR, 3)
        self.w.writerow({
            "lang": src["lang"], "phrase": src["phrase"],
            "halluc_count": src.get("halluc_count", ""),
            "target": tgt["id"], "target_lang": tgt["lang"],
            "source_system": self.source_system, "src_audio": src["abs_path"],
            "audio_filepath": rel, "duration_s": dur,
            "status": "ok" if rel else "fail", "error": error[:200],
        })
        self.fh.flush()
        self.done.add(self.key(src, tgt))
        self.n_ok += bool(rel)
        self.n_fail += (not rel)

    def close(self):
        self.fh.close()
        self.attempt_file.unlink(missing_ok=True)
        print(f"[{self.dir.name}] {self.n_ok} ok, {self.n_fail} failed -> {self.dir}")
