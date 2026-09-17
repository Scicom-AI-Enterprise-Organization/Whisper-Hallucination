#!/usr/bin/env python3
"""Synthesise lexicon phrases with Scicom-intl/Multilingual-Expressive-TTS-1.7B.

Conditioning is by SPEAKER NAME (a token in the prompt), not a reference waveform -- so
"voice profiles" here means picking names from Scicom-intl/ExpressiveSpeech, and the
DNSMOS table is how you rank which names are worth using.

    <|im_start|>{speaker}: {text}<|speech_start|>  ->  <|s_NNN|> tokens  ->  NeuCodec  ->  24 kHz

Output is resampled to 16 kHz so it is directly comparable with the benchmark arms and
scoreable by the same ASR.
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TOK = re.compile(r"<\|s_(\d+)\|>")
SR_OUT = 16000
SR_CODEC = 24000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrases", type=Path, required=True, help="jsonl: {lang, phrase}")
    ap.add_argument("--out", type=Path, default=Path("tts/out/scicom"))
    ap.add_argument("--model", default="Scicom-intl/Multilingual-Expressive-TTS-1.7B")
    ap.add_argument("--codec-repo", default="Scicom-intl/neucodec")
    ap.add_argument("--speakers", nargs="+",
                    default=["multilingual-tts_audio_Grace", "multilingual-tts_audio_Rahman",
                             "DisfluencySpeech"])
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os
    import nf4_shim; nf4_shim.install()
    import load_neucodec
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    codec = load_neucodec.load(args.codec_repo, token=os.environ.get("HF_TOKEN"))
    codec = codec.eval().to(args.device)

    wav_dir = args.out / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in args.phrases.open(encoding="utf-8")]
    rows, t0 = [], time.time()

    for i, it in enumerate(items):
        spk = args.speakers[i % len(args.speakers)]
        prompt = f"<|im_start|>{spk}: {it['phrase']}<|speech_start|>"
        inp = tok(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature,
                                 repetition_penalty=args.repetition_penalty)
        dec = tok.decode(out[0], skip_special_tokens=False)
        tail = dec.split("<|speech_start|>")
        codes = [int(x) for x in TOK.findall(tail[1])] if len(tail) > 1 else []
        stem = f"{it['lang']}_{i:04d}"
        if not codes:
            rows.append({**it, "speaker": spk, "audio_filepath": "", "n_codes": 0,
                         "duration_s": 0.0, "status": "no_speech_tokens"})
            continue
        with torch.no_grad():
            wav = codec.decode_code(torch.tensor(codes)[None, None].to(args.device))
        x = wav[0, 0].float().cpu().numpy()
        x = resample_poly(x, SR_OUT, SR_CODEC)
        peak = float(np.abs(x).max())
        if peak > 0:
            x = np.clip(x / peak * 0.707, -1, 1)
        sf.write(wav_dir / f"{stem}.flac", x.astype("float32"), SR_OUT, format="FLAC", subtype="PCM_16")
        rows.append({**it, "speaker": spk, "audio_filepath": f"wav/{stem}.flac",
                     "n_codes": len(codes), "duration_s": round(len(x) / SR_OUT, 3),
                     "status": "ok"})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    man = args.out / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"scicom: {ok}/{len(rows)} synthesised, {sum(r['duration_s'] for r in rows)/60:.1f} min -> {man}")


if __name__ == "__main__":
    main()
