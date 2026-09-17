#!/usr/bin/env python3
"""Synthesise lexicon phrases with k2-fsa OmniVoice.

Why it matters here: 646 tagged languages (93/100 of our lexicon, 98.9% of phrases, vs
82/100 for Higgs v3), Apache-2.0, and only 3.3 GB. Coverage and licence are exactly the two
constraints the multilingual positive set runs into.

Runs in its own venv (`.venv_omni`) — the package pins its own torch, which would disturb
both the benchmark venv and the Higgs v3 one.

Mode is **auto** (no reference audio, no instruct), matching what the other candidates were
given: text only, so the comparison measures multilingual intelligibility rather than
voice-cloning fidelity. `language` IS passed, since the lexicon records it and withholding
it would handicap the model for no reason.
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SR_OUT = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrases", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tts/out/omnivoice"))
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-step", type=int, default=32, help="diffusion steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    # OmniVoice ids are mostly ISO-639-3 (`arb`, `npi`, `fil`) while the lexicon uses
    # Whisper's 2-letter codes. An unmapped code does not error - it silently drops to
    # language-agnostic mode, which is a quiet quality loss. See tts/omnivoice_langmap.py.
    alias_path = Path("tts/omnivoice_lang_alias.json")
    alias = json.loads(alias_path.read_text()) if alias_path.exists() else {}

    torch.manual_seed(args.seed)
    model = OmniVoice.from_pretrained(args.model)
    model = model.to(args.device).eval() if hasattr(model, "to") else model
    gcfg = OmniVoiceGenerationConfig(num_step=args.num_step)

    sr_model = None
    for obj, attr in ((model, "sample_rate"), (getattr(model, "config", None), "sample_rate"),
                      (getattr(model, "config", None), "sampling_rate")):
        if obj is not None and getattr(obj, attr, None):
            sr_model = int(getattr(obj, attr)); break
    sr_model = sr_model or 24000
    print(f"loaded; sample_rate={sr_model}", flush=True)

    wav_dir = args.out / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in args.phrases.open(encoding="utf-8")]
    rows, t0 = [], time.time()

    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        def gen(items_):
            with torch.no_grad():
                return model.generate(
                    text=[c["phrase"] for c in items_],
                    language=[alias.get(c["lang"], c["lang"]) for c in items_],
                    generation_config=gcfg,
                )

        # A single bad phrase raises for the WHOLE batch ("zero-size array to reduction"),
        # which cost 8 clips per failure on the first pass. So on a batch error, fall back
        # to one-at-a-time so only the genuinely bad phrase is lost.
        try:
            waves = list(gen(chunk))
        except Exception as e:
            if start == 0:
                print(f"batch failed ({type(e).__name__}); falling back to per-item", flush=True)
            waves = []
            for c in chunk:
                try:
                    waves.append(gen([c])[0])
                except Exception as e2:
                    waves.append(np.zeros(0, dtype="float32"))
                    rows.append({**c, "audio_filepath": "", "duration_s": 0.0,
                                 "status": f"error:{type(e2).__name__}:{str(e2)[:70]}"})
            # Items already recorded as failures must not be emitted twice below.
            failed = {id(c) for c, w in zip(chunk, waves) if np.asarray(w).size == 0}
            chunk = [c for c in chunk if id(c) not in failed]
            waves = [w for w in waves if np.asarray(w).size > 0]
            if not chunk:
                print(f"  {min(start+args.batch_size,len(items))}/{len(items)} (all failed)", flush=True)
                continue

        for j, (c, w) in enumerate(zip(chunk, waves)):
            i = start + j
            stem = f"{c['lang']}_{i:04d}"
            x = np.asarray(w, dtype="float32").squeeze()
            if x.size < SR_OUT // 20:
                rows.append({**c, "audio_filepath": "", "duration_s": 0.0, "status": "too_short"})
                continue
            if sr_model != SR_OUT:
                from math import gcd
                g = gcd(sr_model, SR_OUT)
                x = resample_poly(x, SR_OUT // g, sr_model // g)
            peak = float(np.abs(x).max())
            if peak > 0:
                x = np.clip(x / peak * 0.707, -1, 1)
            sf.write(wav_dir / f"{stem}.flac", x.astype("float32"), SR_OUT,
                     format="FLAC", subtype="PCM_16")
            rows.append({**c, "audio_filepath": f"wav/{stem}.flac",
                         "duration_s": round(len(x) / SR_OUT, 3), "status": "ok"})
        print(f"  {min(start+args.batch_size,len(items))}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    man = args.out / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"omnivoice: {ok}/{len(rows)} synthesised in {time.time()-t0:.0f}s -> {man}")
    if ok < len(rows):
        from collections import Counter
        print("statuses:", dict(Counter(r["status"][:44] for r in rows)))


if __name__ == "__main__":
    main()
