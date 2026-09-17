#!/usr/bin/env python3
"""Harvest the genuine-speech (positive) arm from Lingua Libre via Wikimedia Commons.

Why this source: Lingua Libre is native speakers recording single words and short phrases,
so the transcript is exact by construction and the clip is SHORT and ISOLATED -- the same
acoustic shape as a hallucination. That makes it the true counterpart to the `silence` arm:
audio where `Terima kasih.` is the CORRECT output, not a fabrication.

Licences vary per speaker (~60% CC0, ~40% CC BY-SA 4.0, ~4% CC BY 4.0), so every row
carries its own `license`, `author` and `source_url`. Nothing is relicensed.

  python scripts/harvest_lingualibre.py --langs msa ind --out audio/genuine
  python scripts/harvest_lingualibre.py --all

Resumable: files already on disk are skipped.
"""
import argparse, json, re, sys, time, unicodedata, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "scicom-whisper-hallucination/1.0 (husein.zolkepli@scicom.com.my)"}
SR = 16000

# Lingua Libre language QIDs. Malaysian-context languages first, then a spread for the
# scale-out beyond ms/en/zh/ta.
LANGS = {
    "msa": ("Q9237", "ms", "Malay", 0),
    "ind": ("Q9240", "id", "Indonesian", 0),
    "tam": ("Q5885", "ta", "Tamil", 2500),
    "cmn": ("Q9192", "zh", "Mandarin", 2500),
    "eng": ("Q1860", "en", "English", 2500),
    "fra": ("Q150", "fr", "French", 900),
    "spa": ("Q1321", "es", "Spanish", 900),
    "deu": ("Q188", "de", "German", 900),
    "por": ("Q5146", "pt", "Portuguese", 900),
    "ita": ("Q652", "it", "Italian", 900),
    "rus": ("Q7737", "ru", "Russian", 900),
    "jpn": ("Q5287", "ja", "Japanese", 900),
    "kor": ("Q9176", "ko", "Korean", 900),
    "hin": ("Q1568", "hi", "Hindi", 900),
    "ara": ("Q13955", "ar", "Arabic", 900),
    "vie": ("Q9199", "vi", "Vietnamese", 900),
    "tha": ("Q9217", "th", "Thai", 900),
    "nld": ("Q7411", "nl", "Dutch", 900),
    "pol": ("Q809", "pl", "Polish", 900),
    "tur": ("Q256", "tr", "Turkish", 900),
}
# Trailing "(B)", "(SV)" etc. are Lingua Libre homograph markers, not spoken content.
VARIANT = re.compile(r"\s*\([A-Z]{1,3}\)\s*$")


def api(tries=6, **p):
    p.update(format="json", formatversion=2)
    url = API + "?" + urllib.parse.urlencode(p)
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if i + 1 < tries and (code is None or code in (429, 500, 502, 503)):
                time.sleep(2.5 * (i + 1))
                continue
            raise


def list_files(qid: str, code: str, cap: int) -> list[str]:
    prefix, out, cont = f"LL-{qid} ({code})-", [], None
    while True:
        p = dict(action="query", list="allimages", aiprefix=prefix, ailimit=500)
        if cont:
            p["aicontinue"] = cont
        d = api(**p)
        out += [i["name"] for i in d.get("query", {}).get("allimages", [])]
        cont = d.get("continue", {}).get("aicontinue")
        if not cont or (cap and len(out) >= cap):
            break
        time.sleep(0.4)
    return out[:cap] if cap else out


def fetch_meta(names: list[str]) -> dict[str, dict]:
    """Batched imageinfo: download URL plus the attribution fields the licences require."""
    meta = {}
    strip = re.compile(r"<[^>]+>")
    for i in range(0, len(names), 50):
        batch = "|".join("File:" + n for n in names[i:i + 50])
        d = api(action="query", titles=batch, prop="imageinfo",
                iiprop="url|size|extmetadata", iiextmetadatafilter="LicenseShortName|Artist")
        for page in d.get("query", {}).get("pages", []):
            ii = (page.get("imageinfo") or [{}])[0]
            if not ii.get("url"):
                continue
            em = ii.get("extmetadata") or {}
            author = strip.sub("", (em.get("Artist") or {}).get("value", "")).strip()
            author = re.sub(r"\s+", " ", author)
            meta[page["title"].removeprefix("File:").replace(" ", "_")] = {
                "url": ii["url"],
                "license": (em.get("LicenseShortName") or {}).get("value", "unknown"),
                "author": author[:300],
            }
        time.sleep(0.35)
    return meta


def parse_name(name: str, qid: str, code: str) -> tuple[str, str] | None:
    m = re.match(rf"LL-{re.escape(qid)}_\({code}\)-(.+)\.(wav|ogg|flac|mp3)$", name)
    if not m:
        return None
    rest = m.group(1)
    # Speaker names never contain "-", but words do ("sama-sama"), so split on the first only.
    if "-" not in rest:
        return None
    speaker, word = rest.split("-", 1)
    text = VARIANT.sub("", word.replace("_", " ")).strip()
    return (speaker.replace("_", " "), text) if text else None


def to_flac(raw: bytes, dest: Path) -> float:
    import io
    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x = x.mean(axis=1)                                  # to mono
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    peak = float(np.abs(x).max())
    if peak > 0:                                        # normalise to a consistent -3 dBFS
        x = x / peak * 0.707
    x = np.clip(x, -1.0, 1.0).astype("float32")
    sf.write(dest, x, SR, format="FLAC", subtype="PCM_16")
    return len(x) / SR


def grab(item, wav_dir: Path):
    name, info, speaker, text = item
    dest = wav_dir / (Path(name).stem + ".flac")
    if dest.exists():
        try:
            return {"skipped": True, "dur": sf.info(dest).duration, "dest": dest}
        except Exception:
            dest.unlink(missing_ok=True)
    for i in range(4):
        try:
            raw = urllib.request.urlopen(
                urllib.request.Request(info["url"], headers=UA), timeout=120).read()
            return {"dur": to_flac(raw, dest), "dest": dest}
        except Exception as e:
            if i == 3:
                return {"error": f"{type(e).__name__}: {str(e)[:70]}"}
            time.sleep(1.5 * (i + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", default=["msa", "ind"], choices=list(LANGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "audio" / "genuine")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    codes = list(LANGS) if args.all else args.langs
    wav_dir = args.out / "flac"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"

    import csv
    rows = []
    if manifest_path.exists():
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
        print(f"[resume] {len(rows)} rows already in manifest")
    have = {r["audio_filepath"] for r in rows}

    for code in codes:
        qid, iso, label, cap = LANGS[code]
        print(f"\n=== {label} ({code} / {iso}){f', cap {cap}' if cap else ''}")
        names = list_files(qid, code, cap)
        parsed = {}
        for n in names:
            p = parse_name(n, qid, code)
            if p:
                parsed[n] = p
        print(f"  listed {len(names)}, parsed {len(parsed)}")
        meta = fetch_meta(list(parsed))
        print(f"  metadata for {len(meta)}")

        work = [(n, meta[n], *parsed[n]) for n in parsed if n in meta]
        work = [w for w in work if f"flac/{Path(w[0]).stem}.flac" not in have]
        print(f"  downloading {len(work)}")
        ok = err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(grab, w, wav_dir): w for w in work}
            for i, fut in enumerate(as_completed(futs), 1):
                w = futs[fut]
                res = fut.result()
                if not res or "error" in (res or {}):
                    err += 1
                    continue
                ok += 1
                rows.append({
                    "audio_filepath": f"flac/{res['dest'].name}",
                    "lang": iso, "ll_lang_code": code,
                    "text": w[3], "speaker": w[2],
                    "duration_s": round(res["dur"], 3),
                    "license": w[1]["license"], "author": w[1]["author"],
                    "source_url": w[1]["url"],
                })
                if i % 250 == 0:
                    print(f"    {i}/{len(work)} ok={ok} err={err}", flush=True)
        print(f"  done: ok={ok} err={err}")

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["audio_filepath", "lang", "ll_lang_code", "text",
                                              "speaker", "duration_s", "license", "author", "source_url"])
            w.writeheader(); w.writerows(rows)

    import collections
    print(f"\ntotal {len(rows)} clips, {sum(float(r['duration_s']) for r in rows)/3600:.2f} h")
    print("by lang:", dict(collections.Counter(r["lang"] for r in rows).most_common()))
    print("by licence:", dict(collections.Counter(r["license"] for r in rows)))
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
