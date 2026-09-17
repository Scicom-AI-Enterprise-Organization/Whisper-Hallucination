#!/usr/bin/env python3
"""OpenVoice v2 -- tone-colour conversion, and the same converter behind MeloTTS for cloning.

Two modes, because OpenVoice v2 is not one model:
  --mode vc     convert an existing clip's timbre to the target      (any source language)
  --mode clone  MeloTTS renders the phrase, then the converter re-voices it to the target
                (MeloTTS covers en/es/fr/zh/jp/kr only -- that limit is the finding)

`se_extractor` is bypassed: it runs whisper-timestamped to chop a reference into speech
segments, which we do not need -- our references are already clean single-speaker clips, so
`extract_se` takes them directly. Watermarking is off; it perturbs the audio the scorer
then measures, and nothing here is being published as a voice.

    .venv_openvoice/bin/python tts/vc_openvoice.py --mode vc --device cuda:6
"""
import argparse, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vc_common import ROOT, Writer, load_sources, load_targets  # noqa: E402

MELO_LANGS = {"en": "EN", "es": "ES", "fr": "FR", "zh": "ZH", "ja": "JP", "ko": "KR"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["vc", "clone"], default="vc")
    ap.add_argument("--sources", type=Path, default=ROOT / "tts" / "out" / "omnivoice")
    ap.add_argument("--targets", type=Path, default=ROOT / "tts" / "vc_targets")
    ap.add_argument("--out", type=Path, default=ROOT / "tts" / "vc_out")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "vc_repos" / "OpenVoice" / "checkpoints_v2")
    ap.add_argument("--repo", type=Path, default=ROOT / "vc_repos" / "OpenVoice")
    ap.add_argument("--limit", type=int, default=0, help="first N source clips only (smoke test)")
    ap.add_argument("--ref", choices=["short", "long"], default="short",
                    help="one-shot reference: 6 s (the realistic case) or the full ~20 s clip")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))
    from openvoice.api import ToneColorConverter

    # `enable_watermark=False` is documented but this build forwards **kwargs straight to a
    # parent that does not accept it. Disarming the model afterwards has the same effect:
    # `add_watermark` returns the audio untouched when there is no model.
    conv = ToneColorConverter(str(args.ckpt / "converter" / "config.json"), device=args.device)
    conv.watermark_model = None
    conv.load_ckpt(str(args.ckpt / "converter" / "checkpoint.pth"))
    out_sr = conv.hps.data.sampling_rate

    sources, targets = load_sources(args.sources), load_targets(args.targets)
    if args.limit:
        sources = sources[:args.limit]
    system = "openvoice" if args.mode == "vc" else "openvoice_clone"
    w = Writer(args.out, system, args.sources.name if args.mode == "vc" else "melotts")
    print(f"[{system}] {len(sources)} phrases x {len(targets)} targets @ {out_sr} Hz", flush=True)

    melo = {}
    if args.mode == "clone":
        from melo.api import TTS
        for code in sorted({MELO_LANGS[s["lang"]] for s in sources if s["lang"] in MELO_LANGS}):
            melo[code] = TTS(language=code, device=args.device)

    refkey = "ref_path" if args.ref == "long" else "ref_short_path"
    ses = {t["id"]: conv.extract_se([t[refkey]]) for t in targets}
    tmp = args.out / system / "_tmp.wav"

    for t in targets:
        for i, s in enumerate(sources):
            try:
                if args.mode == "clone":
                    code = MELO_LANGS.get(s["lang"])
                    if code is None:
                        w.add(s, t, None, out_sr, i, error=f"MeloTTS has no voice for '{s['lang']}'")
                        continue
                    tts = melo[code]
                    spk = list(tts.hps.data.spk2id.values())[0]
                    tts.tts_to_file(s["phrase"], spk, str(tmp), speed=1.0)
                    src_path = str(tmp)
                else:
                    src_path = s["abs_path"]
                src_se = conv.extract_se([src_path])
                audio = conv.convert(audio_src_path=src_path, src_se=src_se, tgt_se=ses[t["id"]])
                w.add(s, t, audio, out_sr, i)
            except Exception as e:
                traceback.print_exc()
                w.add(s, t, None, out_sr, i, error=f"{type(e).__name__}: {e}")
    w.close()


if __name__ == "__main__":
    main()
