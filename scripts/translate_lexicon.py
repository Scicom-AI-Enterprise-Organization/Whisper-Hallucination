#!/usr/bin/env python3
"""Translate the genuinely-sayable English hallucination phrases into the other languages.

Why this is a *different artefact* from the rest of the lexicon, and labelled as such:

  The lexicon records phrases Whisper was OBSERVED to hallucinate. A translation of
  `thank you` into Tamil is not an observed Tamil hallucination -- it is a phrase Tamil
  speakers genuinely say. That makes it a good POSITIVE example (the contrastive pool needs
  "things people really say"), but it is NOT evidence about what Whisper invents in Tamil,
  and it must never be fed to `ban_candidates` as though it were. Rows are written with
  `source=translated:en` so the distinction survives into the manifest.

Selection matters as much as translation. The top English entries are a mix of real
utterances (`thank you`, `i m sorry`), bare function words (`the`, `uh`, `a`) and sound-effect
labels (`meow`, `beeping`). Only the first kind survives translation meaningfully, so
single-token entries and a sound-effect stoplist are dropped.

    .venv_bench/bin/python scripts/translate_lexicon.py --min-count 50 --max-phrases 60
"""
import argparse, csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ISO-639-1 -> NLLB (FLORES-200) code. Only languages our TTS can actually speak.
NLLB = {
    "ms": "zsm_Latn", "id": "ind_Latn", "ta": "tam_Taml", "zh": "zho_Hans", "ar": "arb_Arab",
    "hi": "hin_Deva", "ja": "jpn_Jpan", "ko": "kor_Hang", "th": "tha_Thai", "vi": "vie_Latn",
    "tr": "tur_Latn", "ru": "rus_Cyrl", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn",
    "pt": "por_Latn", "it": "ita_Latn", "nl": "nld_Latn", "pl": "pol_Latn", "sv": "swe_Latn",
    "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn", "el": "ell_Grek", "he": "heb_Hebr",
    "ur": "urd_Arab", "bn": "ben_Beng", "te": "tel_Telu", "mr": "mar_Deva", "gu": "guj_Gujr",
    "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru", "ne": "npi_Deva", "si": "sin_Sinh",
    "km": "khm_Khmr", "lo": "lao_Laoo", "my": "mya_Mymr", "fa": "pes_Arab", "sw": "swh_Latn",
    "yo": "yor_Latn", "ha": "hau_Latn", "am": "amh_Ethi", "cs": "ces_Latn", "sk": "slk_Latn",
    "sl": "slv_Latn", "hr": "hrv_Latn", "sr": "srp_Cyrl", "bg": "bul_Cyrl", "ro": "ron_Latn",
    "hu": "hun_Latn", "uk": "ukr_Cyrl", "lt": "lit_Latn", "lv": "lvs_Latn", "et": "est_Latn",
    "mt": "mlt_Latn", "ca": "cat_Latn", "eu": "eus_Latn", "gl": "glg_Latn", "is": "isl_Latn",
    "cy": "cym_Latn", "af": "afr_Latn", "az": "azj_Latn", "kk": "kaz_Cyrl", "uz": "uzn_Latn",
    "ka": "kat_Geor", "hy": "hye_Armn", "mn": "khk_Cyrl", "tl": "tgl_Latn", "sq": "als_Latn",
    "mk": "mkd_Cyrl", "be": "bel_Cyrl", "sd": "snd_Arab", "ps": "pbt_Arab", "tg": "tgk_Cyrl",
    "jw": "jav_Latn", "su": "sun_Latn", "mg": "plt_Latn", "sn": "sna_Latn", "so": "som_Latn",
}

# Whisper emits these as sound-effect tags, not speech; a translation of "meow" is noise.
SOUND_TAGS = {"woof", "meow", "beeping", "beep", "bang", "music", "applause", "laughter",
              "silence", "clicking", "ringing", "buzzing", "coughing", "sneeze", "whistle"}


def sayable(phrase: str) -> bool:
    """Is this an utterance a person could say, rather than a token or a sound label?"""
    toks = phrase.split()
    if len(toks) < 2:
        return False                      # 'the', 'uh', 'so' -- no meaningful translation
    if phrase.strip().lower() in SOUND_TAGS:
        return False
    if any(t in SOUND_TAGS for t in toks) and len(toks) <= 2:
        return False
    if re.search(r"\d", phrase):
        return False                      # digits round-trip badly through TTS+ASR
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=ROOT / "lexicon" / "combined_lexicon.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "lexicon" / "translated_lexicon.csv")
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--max-phrases", type=int, default=60)
    ap.add_argument("--langs", nargs="*", default=None, help="default: every language in NLLB map")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.lexicon.open(encoding="utf-8"))
            if r["lang"] == "en" and int(r.get("count") or 0) >= args.min_count]
    phrases = [r for r in rows if sayable(r["phrase"])]
    phrases.sort(key=lambda r: -int(r["count"]))
    phrases = phrases[:args.max_phrases]
    targets = args.langs or sorted(NLLB)
    print(f"{len(rows)} en phrases over count>={args.min_count}; {len(phrases)} sayable; "
          f"{len(targets)} target languages -> {len(phrases) * len(targets)} translations", flush=True)
    for r in phrases[:8]:
        print(f"   keep: {r['phrase']!r}")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device).eval()

    out_rows = []
    for lang in targets:
        code = NLLB.get(lang)
        if not code:
            continue
        try:
            bos = tok.convert_tokens_to_ids(code)
            texts = [r["phrase"] for r in phrases]
            hyps = []
            for i in range(0, len(texts), args.batch_size):
                chunk = texts[i:i + args.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=128).to(args.device)
                with torch.no_grad():
                    gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=96,
                                         num_beams=4)
                hyps += tok.batch_decode(gen, skip_special_tokens=True)
            for r, h in zip(phrases, hyps):
                h = h.strip()
                if not h or h.lower() == r["phrase"].lower():
                    continue          # untranslated passthrough is not a new phrase
                out_rows.append({"phrase_norm": h.lower(), "phrase": h, "lang": lang,
                                 "count": r["count"], "source": "translated:en",
                                 "source_phrase": r["phrase"]})
            print(f"  {lang} ({code}): {len(hyps)} translated", flush=True)
        except Exception as e:
            print(f"  {lang} ({code}) FAILED: {type(e).__name__}: {e}", flush=True)

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["phrase_norm", "phrase", "lang", "count",
                                           "source", "source_phrase"])
        w.writeheader(); w.writerows(out_rows)
    langs = {r["lang"] for r in out_rows}
    print(f"\n{len(out_rows)} translated phrases across {len(langs)} languages -> {args.out}")


if __name__ == "__main__":
    main()
