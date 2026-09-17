#!/usr/bin/env python3
"""Resolve lexicon language codes to OmniVoice's IDs, and report what failed.

OmniVoice exposes 646 ids that are mostly ISO-639-3 (`arb`, `npi`, `tgl`) while the lexicon
uses Whisper's ISO-639-1-ish codes (`ar`, `ne`, `tl`). Unmapped codes silently fall back to
"language-agnostic mode", which is a quiet quality loss rather than an error - so resolve
them explicitly and write the map out for the synthesiser to use.
"""
import collections
import csv
import json
from pathlib import Path

from omnivoice import OmniVoice

CAND = {
    "ar": ["arb", "apc", "arz"], "ne": ["npi", "nep"], "la": ["lat"],
    "mg": ["plt", "mlg"], "fo": ["fao"], "jw": ["jav", "jv"],
    "tl": ["tgl", "fil"], "su": ["sun"], "yue": ["yue", "zh"],
}

m = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
ids = m.supported_language_ids()
print(f"OmniVoice ids: {len(ids)}")

resolved = {}
for k, opts in CAND.items():
    hit = next((o for o in opts if o in ids), None)
    resolved[k] = hit
    print(f"  {k} -> {hit}")

alias = {k: v for k, v in resolved.items() if v}
Path("tts/omnivoice_lang_alias.json").write_text(json.dumps(alias, indent=2))

rows = list(csv.DictReader(open("lexicon/combined_lexicon.csv", encoding="utf-8")))
lex = collections.Counter(r["lang"] for r in rows)
mapped = lambda l: alias.get(l, l)
cov = {l for l in lex if mapped(l) in ids}
ph = sum(lex[l] for l in cov)
print(f"\nwith aliases: {len(cov)}/{len(lex)} languages, {ph}/{len(rows)} phrases ({ph / len(rows):.1%})")
absent = sorted([l for l in lex if mapped(l) not in ids], key=lambda l: -lex[l])
print("still absent:", ", ".join(f"{l}({lex[l]})" for l in absent) or "none")

man = Path("tts/out/omnivoice/manifest.csv")
if man.exists():
    print("\nfailures from the last run:")
    for x in csv.DictReader(man.open(encoding="utf-8")):
        if x["status"] != "ok":
            print(f"  [{x['lang']}] {x['status'][:34]}  phrase={x['phrase'][:40]!r}")
