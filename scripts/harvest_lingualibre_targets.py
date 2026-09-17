#!/usr/bin/env python3
"""Harvest ISOLATED genuine recordings of the high-risk phrases from Lingua Libre.

Why these matter more than their count suggests: a hallucination surfaces as a bare
`Terima kasih.` and nothing else. A Lingua Libre clip is a native speaker saying exactly
that one phrase with nothing around it -- so it is the true minimal pair. Corpus clips
(build_genuine_arm.py) have the phrase buried mid-sentence, which is an easier case.

Wikimedia throttles bots hard (HTTP 429 at 6 concurrent workers), so this is deliberately
SERIAL with a delay between requests. The target set is small, so that is affordable.
Per-file licence and attribution are recorded; nothing is relicensed.
"""
import argparse, csv, io, json, re, time, unicodedata, urllib.error, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "scicom-whisper-hallucination/1.0 (husein.zolkepli@scicom.com.my)"}
SR = 16000
DELAY = 1.1          # be a good citizen; Wikimedia asks bots to stay slow

LANGS = {  # Lingua Libre code -> (QID, iso)
    "msa": ("Q9237", "ms"), "ind": ("Q9240", "id"), "eng": ("Q1860", "en"),
    "cmn": ("Q9192", "zh"), "tam": ("Q5885", "ta"), "fra": ("Q150", "fr"),
    "spa": ("Q1321", "es"), "deu": ("Q188", "de"), "por": ("Q5146", "pt"),
    "ita": ("Q652", "it"), "rus": ("Q7737", "ru"), "jpn": ("Q5287", "ja"),
    "nld": ("Q7411", "nl"), "pol": ("Q809", "pl"), "tur": ("Q256", "tr"),
}
VARIANT = re.compile(r"\s*\([A-Z]{1,3}\)\s*$")
_PUNCT = re.compile(r"[.?!,;:\"'`´()\[\]…]+")
# CC BY-SA is copyleft; keep it but flag it so downstream can filter.
ALLOWED = {"CC0", "CC BY 4.0", "CC BY 3.0", "CC BY-SA 4.0", "CC BY-SA 3.0"}


def norm(t):
    t = unicodedata.normalize("NFC", t or "").strip().casefold()
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", t)).strip()


def api(tries=6, **p):
    p.update(format="json", formatversion=2)
    url = API + "?" + urllib.parse.urlencode(p)
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if i + 1 < tries:
                time.sleep(5 * (i + 1)); continue
            raise


def list_lang(qid, code, cache: Path):
    """Enumerate one language's Lingua Libre filenames, cached to disk."""
    cf = cache / f"ll_{code}_names.json"
    if cf.exists():
        return json.loads(cf.read_text())
    out, cont = [], None
    while True:
        p = dict(action="query", list="allimages", aiprefix=f"LL-{qid} ({code})-", ailimit=500)
        if cont:
            p["aicontinue"] = cont
        d = api(**p)
        out += [i["name"] for i in d.get("query", {}).get("allimages", [])]
        cont = d.get("continue", {}).get("aicontinue")
        if not cont:
            break
        time.sleep(DELAY)
    cf.write_text(json.dumps(out))
    return out


def to_flac(raw, dest):
    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    peak = float(np.abs(x).max())
    if x.size < SR // 20 or peak <= 0:
        return None
    sf.write(dest, np.clip(x / peak * 0.707, -1, 1).astype("float32"), SR,
             format="FLAC", subtype="PCM_16")
    return len(x) / SR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "genuine_isolated")
    ap.add_argument("--cache", type=Path, default=ROOT / "build" / "ll_cache")
    ap.add_argument("--langs", nargs="+", default=["msa", "ind", "eng", "cmn", "tam"])
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    flac_dir = args.out / "flac"; flac_dir.mkdir(parents=True, exist_ok=True)

    # Target phrases, plus the short courtesy words that dominate hallucination tallies.
    tgt_by_lang: dict[str, set[str]] = {}
    for r in csv.DictReader((ROOT / "phrases" / "targets.csv").open(encoding="utf-8")):
        tgt_by_lang.setdefault(r["lang"], set()).add(norm(r["phrase"]))
    for lang, extra in {
        "ms": ["terima kasih", "sama-sama", "selamat", "maaf", "tolong", "sila", "ya", "tidak", "baik"],
        "id": ["terima kasih", "sama-sama", "selamat", "maaf", "tolong", "ya", "tidak", "baik"],
        "en": ["thank you", "thanks", "okay", "yeah", "bye", "goodbye", "please", "sorry"],
        "zh": ["谢谢", "感谢", "再见", "好", "是", "对"],
        "ta": ["நன்றி", "வணக்கம்", "சரி", "ஆம்"],
    }.items():
        tgt_by_lang.setdefault(lang, set()).update(norm(x) for x in extra)

    man = args.out / "manifest.csv"
    rows = list(csv.DictReader(man.open(encoding="utf-8"))) if man.exists() else []
    have = {r["audio_filepath"] for r in rows}

    for code in args.langs:
        qid, iso = LANGS[code]
        targets = tgt_by_lang.get(iso, set())
        names = list_lang(qid, code, args.cache)
        want = []
        for n in names:
            m = re.match(rf"LL-{re.escape(qid)}_\({code}\)-(.+)\.(wav|ogg|flac|mp3)$", n)
            if not m or "-" not in m.group(1):
                continue
            speaker, word = m.group(1).split("-", 1)
            text = VARIANT.sub("", word.replace("_", " ")).strip()
            if norm(text) in targets:
                want.append((n, speaker.replace("_", " "), text))
        print(f"\n=== {code}/{iso}: {len(names)} files listed, {len(want)} match a target phrase", flush=True)

        for i in range(0, len(want), 50):
            chunk = want[i:i + 50]
            d = api(action="query", prop="imageinfo", iiprop="url|extmetadata",
                    iiextmetadatafilter="LicenseShortName|Artist",
                    titles="|".join("File:" + c[0] for c in chunk))
            meta = {}
            strip = re.compile(r"<[^>]+>")
            for page in d.get("query", {}).get("pages", []):
                ii = (page.get("imageinfo") or [{}])[0]
                if not ii.get("url"):
                    continue
                em = ii.get("extmetadata") or {}
                meta[page["title"].removeprefix("File:").replace(" ", "_")] = {
                    "url": ii["url"],
                    "license": (em.get("LicenseShortName") or {}).get("value", "unknown"),
                    "author": re.sub(r"\s+", " ", strip.sub("", (em.get("Artist") or {}).get("value", ""))).strip()[:300],
                }
            time.sleep(DELAY)

            for name, speaker, text in chunk:
                info = meta.get(name)
                if not info or info["license"] not in ALLOWED:
                    continue
                stem = Path(name).stem
                rel = f"flac/{stem}.flac"
                if rel in have:
                    continue
                try:
                    raw = urllib.request.urlopen(
                        urllib.request.Request(info["url"], headers=UA), timeout=120).read()
                    dur = to_flac(raw, flac_dir / f"{stem}.flac")
                except Exception as e:
                    print(f"   !! {stem}: {type(e).__name__}", flush=True)
                    time.sleep(4); continue
                time.sleep(DELAY)
                if dur is None:
                    continue
                have.add(rel)
                rows.append({
                    "audio_filepath": rel, "lang": iso, "source": "lingualibre",
                    "text": text, "duration_s": round(dur, 3), "license": info["license"],
                    "has_target_phrase": "true", "target_phrases": norm(text),
                    "note": "isolated single-phrase recording by a native speaker",
                    "author": info["author"], "source_url": info["url"],
                })
            print(f"   {len(rows)} kept so far", flush=True)

        fields = ["audio_filepath", "lang", "source", "text", "duration_s", "license",
                  "has_target_phrase", "target_phrases", "note", "author", "source_url"]
        with man.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    import collections
    print(f"\ntotal {len(rows)} isolated clips")
    print("by lang:", dict(collections.Counter(r["lang"] for r in rows)))
    print("by licence:", dict(collections.Counter(r["license"] for r in rows)))
    print("phrases:", dict(collections.Counter(r["target_phrases"] for r in rows).most_common(12)))


if __name__ == "__main__":
    main()
