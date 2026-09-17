# Sources

Everything gathered, with provenance and licence. Two groups: **hallucination evidence**
(what Whisper wrongly emits) and **Malaysian speech** (where genuine utterances of those
same phrases come from).

## 1. Hallucination phrase lexicons — vendored in `lexicon/`

| file | rows | source | licence |
|---|---:|---|---|
| `agh_boh.csv` | 294 | AGH DSP "Bag of Hallucinations", ICASSP 2025 | MIT |
| `agh_hallucination_list.csv` | 30,407 | same paper, full tally over non-speech | MIT |
| `hf_whisper_hallucinations_phrases.csv` | 7,889 | `sachaarbonel/whisper-hallucinations` | MIT |
| `granary/*.txt` | 3,028 / 23 langs | **NVIDIA NeMo SDP**, Granary pipeline | Apache-2.0 |
| `combined_lexicon.csv` | **40,891** | merged + normalised by `build_lexicon.py` | — |

The NeMo lists ship with `DetectWhisperHallucinationFeatures`
(`sdp/processors/inference/asr/utils/whisper_hallucinations.py`, vendored as a
reference copy). It flags three things: repeated n-grams (unique-word share ≤ 0.4),
absurdly long words, and exact matches against the per-language phrase file.

**Malay coverage in public lexicons is almost nil** — 6 `ms` phrases total, versus
30,443 `en`. The Malaysian side has to be built here, not downloaded.

Top observed English hallucinations: `thank you` (31,353), `thanks for watching`
(25,053), `thank you for watching` (6,264). The Malay ones that do appear are
`terima kasih`, `terima kasih kerana menonton`, `terima kasih banyak banyak`.

## 2. Hallucination-labelled audio

| source | content | licence | notes |
|---|---|---|---|
| **HALAS** (`DSP-AGH/HALAS`, `MatBar99/HALAS`) | 3,611 Earnings-22 clips, span-level labels across **9 ASR models** | CC BY 4.0 | the only public set that separates **Hallucination**, **Looping**, and **Hallucination Looping** |
| **Careless Whisper** (`koenecke/hallucination_harms`) | **187 wav / 42.8 min** of de-identified AphasiaBank audio that actually hallucinated | no licence file (cite the FAccT'24 paper) | real speech with long pauses — the naturalistic trigger |

HALAS per-model "Hallucination or looping" counts out of 3,611 — useful as a prior for
which model to start from:

| model | flagged |
|---|---:|
| whisper-large-v2 | 1,581 |
| parakeet-tdt-v2 | 1,217 |
| canary-1b | 1,213 |
| whisper-large-v3-turbo | 1,060 |
| granite | 1,002 |
| **whisper-large-v3** | **858** |
| CrisperWhisper | 772 |

large-v3-turbo hallucinates *more* than large-v3; CrisperWhisper least of the Whisper family.

## 3. Non-speech audio (negative class)

| corpus | size | licence | verdict |
|---|---|---|---|
| **MUSAN** (openslr.org/17) | ~11 GB noise/music/speech | **CC BY 4.0** | ✅ default — commercially usable |
| ESC-50 | 2,000 clips | CC BY-NC 3.0 | research/eval only |
| UrbanSound8K | 8,732 clips | CC BY-NC | Calm-Whisper measured **99.97%** hallucination rate for large-v3 on it |
| AudioSet / FSD50K | very large | mixed / CC BY 4.0 | used by the ICASSP'25 study |

## 4. Public Malaysian speech — positive class

Full table with licences in `sources/malaysian_speech_registry.csv` (24 datasets).
Highlights:

- `mesolitica/Malaysian-STT-Whisper` — 5.86M rows, ms/en/zh/ta/id, word + segment
  timestamps. Labels are **Whisper pseudo-labels**, so they must be re-verified with the
  STT API before being treated as ground truth.
- `mesolitica/pseudolabel-malaysian-youtube-whisper-large-v3-timestamp` — labelled *by
  large-v3 itself*, so it is a direct mine for **naturally occurring large-v3
  hallucinations** on Malaysian audio, at scale.
- `malaysia-ai/malaysian-podcast-youtube`, `-dialects-youtube` — conversational register,
  where closers like `terima kasih` are actually frequent. CC BY-NC 4.0.
- `SEACrowd/sarawak_malay` (**CC0**), `SaLTUNIMAS/sarawak-malay-asr` (**CC BY 4.0**) —
  the only fully unencumbered Malaysian sets found.
- `google/fleurs` `ms_my` — 3,740 utterances, CC BY 4.0, human-read. **Only 1 contains
  "terima kasih"**: it is read Wikipedia prose, so conversational closers are near-absent.
  Good for the WER regression guard, useless as a positive pool.
- Restricted, flagged and excluded from redistributable outputs:
  `Speech-data/Malay-Speech-Dataset` (CC BY-NC-**ND** — no derivatives, so clips cannot
  be cut from it), plus the NC sets above.
- Behind agreements: IMDA National Speech Corpus (Singapore), MagicHub ASR-MalCSC (5 h
  conversational Malay), ASR-SMalDuSC (4.8 h scripted).

## 5. Papers

- Koenecke et al., *Careless Whisper*, FAccT 2024 — ~1% of AphasiaBank segments hallucinate; long pauses are the trigger.
- Barański et al., *Whisper ASR Hallucinations Induced by Non-Speech Audio*, ICASSP 2025 — the BoH construction.
- Wang et al., *Calm-Whisper*, Interspeech 2025 — 3 attention heads carry most non-speech hallucination; 99.97% → 15.51%.
- *HALAS*, 2026 — the human-annotated looping taxonomy used here.
