#!/usr/bin/env python3
"""Measure the ASR judge's own floor per language, on REAL human speech.

`filter_lexicon_synth.py` accepts a synthetic clip when Whisper can recover the phrase from
it. That silently assumes Whisper CAN transcribe the language. For low-resource languages it
cannot, so a perfectly good clip fails and the corpus ends up biased toward languages the
judge happens to read well -- using Whisper to validate Whisper-targeted positives is
circular exactly where Whisper is weakest.

So measure the floor: run the same judge, same settings, on real human recordings (FLEURS)
in each language. A language whose real-speech CER is already 0.6 cannot be held to a 0.25
gate on synthetic audio, and its low yield says nothing about the TTS.

    .venv_bench/bin/python tts/judge_floor.py --langs lt az ne am my --per-lang 12
"""
import argparse, io, json, statistics, sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import cer  # noqa: E402

SR = 16000
# FLEURS uses locale codes; map our 2-letter codes to them.
FLEURS = {
    "lt": "lt_lt", "az": "az_az", "ne": "ne_np", "am": "am_et", "my": "my_mm",
    "fi": "fi_fi", "sk": "sk_sk", "kn": "kn_in", "th": "th_th", "lo": "lo_la",
    "cs": "cs_cz", "ja": "ja_jp", "pl": "pl_pl", "ta": "ta_in", "ur": "ur_pk",
    "el": "el_gr", "vi": "vi_vn", "ms": "ms_my", "id": "id_id", "en": "en_us",
    "fr": "fr_fr", "de": "de_de", "es": "es_419", "pt": "pt_br", "ar": "ar_eg",
    "zh": "cmn_hans_cn", "ko": "ko_kr", "hi": "hi_in", "km": "km_kh", "sw": "sw_ke",
    "si": "si_lk", "sd": "sd_in", "te": "te_in", "ml": "ml_in", "mg": "mg_mg",
    "yo": "yo_ng", "pa": "pa_in", "gu": "gu_in", "mr": "mr_in", "bn": "bn_in",
    "ka": "ka_ge", "hy": "hy_am", "kk": "kk_kz", "mn": "mn_mn", "ps": "ps_af",
    "sn": "sn_zw", "so": "so_so", "ha": "ha_ng", "yue": "yue_hant_hk", "uz": "uz_uz",
    "sl": "sl_si", "et": "et_ee", "ro": "ro_ro", "da": "da_dk", "bg": "bg_bg",
    "sv": "sv_se", "hu": "hu_hu", "nl": "nl_nl", "it": "it_it", "tr": "tr_tr",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--per-lang", type=int, default=12)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--dataset", default="google/fleurs")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "judge_floor.json")
    args = ap.parse_args()

    from datasets import Audio, load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(args.model)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16).to(args.device).eval()

    out = {}
    for lang in args.langs:
        code = FLEURS.get(lang)
        if not code:
            print(f"[skip] no FLEURS code for {lang}"); continue
        try:
            # datasets>=5 demands torchcodec the moment an audio column is decoded, so hand
            # back raw bytes and decode with soundfile instead (CLAUDE.md).
            ds = load_dataset(args.dataset, code, split="test", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            cers = []
            for i, r in enumerate(ds):
                if i >= args.per_lang:
                    break
                a = r["audio"]
                raw = a["bytes"] if isinstance(a, dict) and a.get("bytes") else None
                if raw is None:
                    x, sr0 = sf.read(a["path"], dtype="float32")
                else:
                    x, sr0 = sf.read(io.BytesIO(raw), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr0 != SR:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g = gcd(int(sr0), SR)
                    x = resample_poly(x, SR // g, int(sr0) // g)
                ref = (r.get("transcription") or r.get("raw_transcription") or "").strip()
                if not ref:
                    continue
                f = proc(x, sampling_rate=SR, return_tensors="pt")
                with torch.no_grad():
                    g = asr.generate(f.input_features.to(args.device, torch.float16),
                                     language=lang, task="transcribe",
                                     max_new_tokens=200, num_beams=1)
                hyp = proc.batch_decode(g, skip_special_tokens=True)[0].strip()
                cers.append(cer(ref, hyp))
            if cers:
                out[lang] = {"n": len(cers),
                             "median_cer": round(statistics.median(cers), 4),
                             "mean_cer": round(statistics.mean(cers), 4)}
                print(f"  {lang:<4} real-speech median CER {out[lang]['median_cer']:.3f} "
                      f"(n={len(cers)})", flush=True)
        except Exception as e:
            print(f"  {lang:<4} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)

    args.out.write_text(json.dumps(out, indent=2) + "\n")
    if out:
        bad = {l: v["median_cer"] for l, v in out.items() if v["median_cer"] > 0.25}
        print(f"\nlanguages where the JUDGE itself exceeds the 0.25 gate on real speech: {bad}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
