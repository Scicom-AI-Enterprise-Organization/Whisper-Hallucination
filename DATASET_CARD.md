---
license:
- cc-by-4.0
- cc-by-sa-4.0
- cc0-1.0
- mit
- apache-2.0
task_categories:
- automatic-speech-recognition
- audio-classification
language:
- ms
- id
- en
- zh
- ta
- multilingual
tags:
- whisper
- hallucination
- repetition
- looping
- asr-robustness
- malaysian
- non-speech
pretty_name: Whisper Hallucination and Repetition Probes
size_categories:
- 10K<n<100K
configs:
- config_name: reduplication
  default: true
  data_files:
  - split: test
    path: data/reduplication/test-*.parquet
- config_name: silence
  data_files:
  - split: test
    path: data/silence/test-*.parquet
- config_name: genuine
  data_files:
  - split: test
    path: data/genuine/test-*.parquet
- config_name: genuine_isolated
  data_files:
  - split: test
    path: data/genuine_isolated/test-*.parquet
- config_name: librispeech_test_clean
  data_files:
  - split: test
    path: data/librispeech_test_clean/test-*.parquet
- config_name: music
  data_files:
  - split: test
    path: data/music/test-*.parquet
- config_name: nonspeech
  data_files:
  - split: test
    path: data/nonspeech/test-*.parquet
- config_name: speech_in_noise
  data_files:
  - split: test
    path: data/speech_in_noise/test-*.parquet
- config_name: lexicon
  data_files:
  - split: train
    path: data/lexicon/train-*.parquet
- config_name: ban_candidates
  data_files:
  - split: train
    path: data/ban_candidates/train-*.parquet
- config_name: targets
  data_files:
  - split: train
    path: data/targets/train-*.parquet
- config_name: malaysian_sources
  data_files:
  - split: train
    path: data/malaysian_sources/train-*.parquet
---

# Whisper Hallucination and Repetition Probes

> **This is a BENCHMARK. Every audio config is `test` — do not fine-tune on it.**
> Training on these clips invalidates every number you would then report. Build training
> data separately from the same source corpora, excluding the items listed in
> [`benchmark/exclusions.json`](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/blob/main/benchmark/exclusions.json) in the project
> repo (546 FMA tracks, 1,168 FSD50K ids,
> 2,620 LibriSpeech utterances, and the Malay/Lingua Libre stems).
>
> Structural disjointness is easy here: the benchmark draws FSD50K's **eval** split and FMA
> **shards 0-1**, so training can use FSD50K **dev** (35,676 voice-free clips) and FMA
> **shards 2-12** and never overlap. Malaysian-Emilia (3,727 h, 17,290 genuine
> `terima kasih`) is untouched by this benchmark and is the natural positive pool.

Audio probes and phrase lexicons for measuring the two failure modes of `whisper-large-v3`:

- **hallucination** — text with no phonetic basis in the audio (`Terima kasih.` over silence)
- **repetition / looping** — the decoder emits a unit far more times than it was spoken

**Code, build scripts and design notes:** [github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination)
— arm builders, the benchmark harness, `benchmark/exclusions.json`, the ablation grid and the
training-corpus tooling all live there.

## The problem this is built around

Every high-frequency Whisper hallucination is also a phrase people genuinely say.
`terima kasih` is the most common Malay hallucination **and** the most common thing a
Malaysian call-centre agent says. You cannot fix this with a text blocklist without
deleting real transcriptions.

And silence is not the only trigger. **Music and noise hallucinate at least as readily** —
Calm-Whisper measured large-v3 producing text on 99.97% of UrbanSound8K clips, none of
which contain speech. Energy in the signal is not evidence for words.

So the dataset is **contrastive**, across a spectrum rather than a binary:

| state | audio | Whisper says | correct action |
|---|---|---|---|
| **silence** | room tone, no speech | `Terima kasih.` | drop — 100% invented |
| **music / noise** | a track, no voice | `Terima kasih.` | drop — energy, no evidence |
| **speech in noise** | real speech at low SNR | part real, part invented | the hard case |
| **clean speech** | someone says it | `Terima kasih.` | keep — correct |

A detector that only reads the text cannot tell these apart. That is the point.

## Baseline scores

Five checkpoints × 8 arms × 11,852 clips per model, measured 2026-09-16/17 on 2× NVIDIA H20
with greedy decoding and no temperature fallback: `large-v2`, `large-v3`, `large-v3-turbo`,
and the two Malaysian fine-tunes `mesolitica/malaysian-whisper-large-v2` and
`mesolitica/Malaysian-whisper-large-v3-turbo-v3`.

**Full tables — hallucination rates, repetition runaway, WER, and what each model actually
emits on silence and music — are in the project repo:
[README → Measured baselines](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination#measured-baselines).** Raw numbers: `bench/scores.json`.

Headline: on non-speech, hallucination runs 9.3-98.8% depending on model and arm. For the
three **OpenAI** checkpoints it is **100.0% everywhere** once a bare `"."` counts as output —
every clip produces something. The Malaysian fine-tunes break that: `Malaysian-turbo-v3`
returns a genuinely empty string on 90.6% of voice-free clips. It pays for that on the other
axis, running away on 14.5% of `reduplication` clips against 5.8% for the checkpoint it came
from, and emitting nothing on 58% of them. LibriSpeech test-clean sits at 3.5% for `large-v3`,
matching the published figure — the check that the harness is wired correctly.

Measuring both failure modes on the same clips is the point: ranked on hallucination alone
`Malaysian-turbo-v3` wins by a distance; ranked on raw WER it looks unusable, and ~1% of
looping clips account for most of that gap.

The benchmark harness ([`bench/`](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/tree/main/bench)) ships with this dataset and in the repo.

## Configs

| config | rows | hours | content | ground truth |
|---|---:|---:|---|---|
| `reduplication` | 1,440 | 1.9 | repeated units — CV syllables, vowels, laughter, clicks | exactly `n_repeats` repeats |
| `silence` | 42 | 0.4 | silence / near-silence, 6 floors × 7 durations | empty string |
| `music` | 600 | 4.5 | **real produced music** — Free Music Archive excerpts | empty string |
| `nonspeech` | 1,168 | 2.8 | FSD50K music + environmental noise, label-verified voice-free | empty string |
| `speech_in_noise` | 1,200 | 3.1 | genuine speech mixed with music/noise at 5 SNRs | the real transcript |
| `genuine` | 4,694 | 12.4 | real Malaysian speech, permissively licensed | human transcript |
| `genuine_isolated` | 88 | 0.05 | native speakers saying one phrase, nothing else | the phrase |
| `librispeech_test_clean` | 2,620 | 5.4 | English WER regression guard | human transcript |
| `lexicon` | 40,891 | — | known hallucination phrases, 100 languages | — |
| `ban_candidates` | 40,891 | — | every lexicon phrase, classified safe/unsafe to blocklist | — |
| `targets` | 463 | — | high-risk phrases: hallucinated *and* genuinely said | — |
| `malaysian_sources` | 24 | — | survey of public Malaysian speech datasets | — |

```python
from datasets import load_dataset

# Audio arms are test-only.
red = load_dataset("Scicom-intl/Whisper-Hallucination", "reduplication",   split="test")
sil = load_dataset("Scicom-intl/Whisper-Hallucination", "silence",         split="test")
mus = load_dataset("Scicom-intl/Whisper-Hallucination", "music",           split="test")
sin = load_dataset("Scicom-intl/Whisper-Hallucination", "speech_in_noise", split="test")
gen = load_dataset("Scicom-intl/Whisper-Hallucination", "genuine",         split="test")

# Lookup tables (not splits).
lex = load_dataset("Scicom-intl/Whisper-Hallucination", "lexicon",         split="train")
ban = load_dataset("Scicom-intl/Whisper-Hallucination", "ban_candidates",  split="train")
```

### Decoding the audio

`datasets >= 5.0` raises `ImportError: To support decoding audio data, please install
'torchcodec'` the moment you touch an audio column. Either install it, or skip it entirely
— the payload is 16 kHz mono FLAC that `soundfile` reads directly:

```python
import io, soundfile as sf
from datasets import load_dataset, Audio

ds = load_dataset("Scicom-intl/Whisper-Hallucination", "reduplication", split="test")
ds = ds.cast_column("audio", Audio(decode=False))          # hand back raw bytes

row = ds[0]
x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")   # 16 kHz mono
print(row["id"], row["n_repeats"], row["reference_text"], len(x) / sr)
```

`streaming=True` works the same way, if you would rather not pull 2 GB to look at a few
clips:

```python
st = load_dataset("Scicom-intl/Whisper-Hallucination", "silence", split="test", streaming=True)
st = st.cast_column("audio", Audio(decode=False))
row = next(iter(st))
```


Audio is 16 kHz mono FLAC (lossless, 16-bit), peak-normalised to −3 dBFS.

## Negative arms — what the model invents

### `reduplication`

Motivated by an observed failure: a recording of the syllable `tu` repeated **four**
times decoded as `tu` repeated until the token limit. Once inside a repeated-token
region, Whisper's decoder has no signal for how many repeats remain.

| field | values |
|---|---|
| `pattern` | `cv` (720), `vowel` (360), `laugh` (216), `click` (144) |
| `unit` | `tu ta ka pa da la na bi ko me` / `a i u e o` / `ha he hi` / `tick beep` |
| `n_repeats` | 3, 4, 5, 6, 8, 12 |
| `rate_hz` | 3, 5, 7, 9 units/sec |
| `tail_silence_s` | 0, 2, 8 |

`click` contains no speech at all, which separates **repetition** from **speech** as the
trigger. Nothing in this config is tied to a language.

Score with `n_hyp / n_repeats`: `1.0` correct, `>1` a runaway, `<1` a mitigation that
deleted genuine repetition.

### `silence`

The cleanest arm to read: the reference is the empty string, so *any* output is a
hallucination. `floor` covers `digital_zero`, `dither`, `hiss_-60db`, `hiss_-45db`,
`hum_50hz`, `roomtone` — true digital zero behaves differently from a realistic quiet
mic. Durations straddle the 30 s window boundary (1, 5, 10, 29, 31, 60, 120 s).

### `music` — 600 excerpts, 4.5 h

**Real produced music**, not sound effects: Free Music Archive tracks (commercial-licence
subset, CC BY 4.0, already 16 kHz mono), excerpted at 10 / 30 / 45 s from inside the track
so intros are skipped. This is what actually plays under a call — hold music, a YouTube
backing bed — and it is a different acoustic distribution from isolated instrument samples.

Reference is the empty string: music is not speech, so a transcript is a hallucination.

**Vocals caveat.** FMA ships no instrumental/vocal tag, so some tracks contain singing.
The `vocals` column records `unknown` rather than a guess. Those rows still carry an empty
reference, because the target behaviour for a music bed is to emit nothing — but if you
need *provably* zero words in the audio, use `nonspeech`, where every clip is
label-verified voice-free.

### `nonspeech` — 1,168 clips, 2.8 h

FSD50K clips (CC BY 4.0) carrying **no voice label of any kind** — 583 music /
musical-instrument, 585 environmental and mechanical noise. Any clip labelled `Speech`,
`Singing`, `Human_voice`, `Chatter`, `Laughter` and 20-odd relatives was excluded, so the
empty reference is verified by annotation, not assumed.

This is the strict-ground-truth non-speech arm; `music` is the realistic one.

### `speech_in_noise` — 1,200 clips, 3.1 h

The case the pure arms cannot reach: **genuine speech with a background bed**, where
transcription degrades into fabrication as evidence weakens rather than appearing from
nothing. 240 clips from `genuine` (preferring those carrying a high-risk phrase) mixed
with voice-free FSD50K music or noise at **SNR = +20, +10, +5, 0, −5 dB**.

Reference is the **real transcript**, so this measures WER degradation and hallucination
onset on the same axis. 295 of the 1,200 carry a target phrase — those are where a
hallucinated `terima kasih` and a real one become genuinely confusable.

## Positive arms — what a mitigation must not destroy

Without these, "hallucination rate fell" is unfalsifiable: a decoder that outputs nothing
scores perfectly on the negative arms.

### `genuine` — 4,694 clips, 12.4 h

| source | clips | licence | register |
|---|---:|---|---|
| `SaLTUNIMAS/sarawak-malay-asr` | 1,164 | CC BY 4.0 | Sarawak Malay interviews, human transcripts |
| `emhaihsan/Synth-Manglish` | 2,457 | CC BY 4.0 | Manglish code-switching (**synthetic TTS voice**) |
| `google/fleurs` `ms_my` | 1,073 | CC BY 4.0 | human-read Wikipedia prose |

**59 clips carry a high-risk phrase** (`has_target_phrase = true`) — these are the real
minimal pairs against the negative arms:

| phrase | clips |
|---|---:|
| `sama-sama` | 41 |
| `terima kasih` | 16 |
| `selamat tinggal` | 2 |
| `terima kasih banyak banyak` | 1 |

The rest are the **regression guard**: ordinary speech whose WER must not move when a
mitigation is applied.

### `librispeech_test_clean` — 2,620 clips, 5.4 h

LibriSpeech test-clean (CC BY 4.0), the split the hallucination-mitigation literature
reports its cost on. Included so a WER figure measured here lands on the same axis as
published results rather than only on our Malaysian arms.

Reference point — hallucination-space projection on large-v3 (arXiv:2609.04561):

| variant | ESC-50 HR | LibriSpeech WER clean / other |
|---|---:|---:|
| baseline | 44.25% | 4.06% / 5.87% |
| gated projection | 8.38% | 6.17% / 6.57% |
| always-on projection | 1.50% | **12.95% / 13.13%** |

Suppressing hallucination is not free: always-on nearly eliminates it and **triples clean
WER**. That exchange rate is the reason the positive arms exist.

### `genuine_isolated` — 88 clips

Native speakers from Lingua Libre (via Wikimedia Commons) saying **one phrase and nothing
else**, across ms / id / en / zh / ta — 3 bare `terima kasih`, plus `thanks`, `okay`,
`please`, `sorry`, `bye`, `好`, `sama-sama`, `maaf`, `selamat`, `baik`, `tidak`, `ya`.
Small, but the closest possible match to the hallucination's acoustic shape: a
hallucination surfaces as a bare `Terima kasih.`, and so does one of these clips. In
`genuine` the phrase sits mid-sentence, which is the easier case.

Licences vary per speaker (CC0 ×43, CC BY-SA 4.0 ×42, CC BY 4.0 ×3), so **every row carries
its own `license`, `author` and `source_url`**. Filter `license != "CC BY-SA 4.0"` if you
need to avoid copyleft.

## `ban_candidates` — which phrases you can actually blocklist

The founding problem, answered with measurement instead of intuition. Every lexicon phrase
is crossed against all 7,402 genuine transcripts here (17.8 h) and classified:

| verdict | phrases | meaning |
|---|---:|---|
| `safe_candidate` | 871 | never observed in genuine speech, multi-word, frequently hallucinated |
| `unsafe` | 389 | occurs in genuine speech — banning it destroys real transcripts |
| `weak_evidence` | 39,631 | never observed, but too rare or too short to be confident |

| phrase | verdict | hallucinated | genuine exact / substring |
|---|---|---:|---|
| `thanks for watching` | **safe_candidate** | 25,053 | 0 / 0 |
| `thank you` | **unsafe** | 31,353 | 0 / 13 |
| `terima kasih` | **unsafe** | 1 | **3** / **19** |

Use it in EXACT mode — drop an output only when the whole transcript equals the phrase,
which is the shape a hallucination takes. `genuine_substring` shows what a more aggressive
contains-match would additionally cost.

**Caveat, and it is the important one:** every `ms` / `zh` / `ta` phrase falls into
`weak_evidence`, because the *hallucination counts* come from public lexicons that barely
cover those languages (6 Malay phrases in total). The classifier is sound — English
demonstrates it — but the Malaysian side needs hallucination counts measured locally, which
this dataset's probe arms exist to produce.

## `lexicon`

Four public sources merged, normalised (NFC, casefold, punctuation-stripped) and
deduplicated on `(phrase, lang)`:

| source key | rows in | origin | licence |
|---|---:|---|---|
| `agh_boh` | 294 | AGH DSP "Bag of Hallucinations", ICASSP 2025 | MIT |
| `agh_full` | 30,407 | same paper, full non-speech tally | MIT |
| `hf_noise` | 7,889 | [`sachaarbonel/whisper-hallucinations`](https://huggingface.co/datasets/sachaarbonel/whisper-hallucinations) | MIT |
| `granary` | 3,028 / 23 langs | **NVIDIA NeMo SDP**, Granary pipeline | Apache-2.0 |

Top English entries: `thank you` (31,353), `thanks for watching` (25,053),
`thank you for watching` (6,264). The NVIDIA lists ship alongside a working
`DetectWhisperHallucinationFeatures` processor, vendored at `lexicon_raw/granary/`.

**Coverage is extremely uneven.** English has 30,443 phrases; Malay has **6**. That gap is
why this dataset exists — the Malaysian side is not downloadable, it has to be measured.

## A note on comparing hallucination rates

Published HRs for the same model on the same corpus differ by more than 10x — Calm-Whisper
reports **99.97%** on UrbanSound8K for large-v3, arXiv:2609.04561 reports **76.08%**. The
gap is methodological: the latter counts only output *surviving Whisper's own no-speech
filter* (`no_speech_threshold = 0.6`), the former counts raw output. Always state which you
mean.

It also matters what is in the corpus. arXiv:2609.04561 filtered FSD50K down to clips
labelled neither speech, vocal **nor music** (HR 21.35%). Our `nonspeech` arm deliberately
**keeps** music — filter `kind == "noise"` to approximate their setup, or use the whole arm
plus `music` for the harder, more realistic condition.

## Known limitations

- **The positive arms are small relative to the negative ones**, and the phrase-matched
  subset is 147 clips (59 + 88). That is what genuinely exists under a redistributable
  licence. The larger Malaysian conversational corpora are CC BY-NC and were excluded on
  purpose so this dataset stays shareable.
- **`Synth-Manglish` is TTS, not human speech.** It is a little over half of `genuine`.
  Filter on `source` if that matters for your claim.
- **`genuine` clips are long** (median 9.6 s) while hallucinations happen on short,
  low-evidence audio. The isolated arm is the short-clip counterpart.
- **`music` may contain singing.** See the vocals caveat above; `nonspeech` is the
  label-verified alternative.
- **`speech_in_noise` is synthetic mixing**, not natively recorded noisy speech. SNR is
  exact and controllable, which is the point, but channel effects and Lombard speech
  (people talk differently in noise) are absent.
- **No ablation results here.** See the design notes in the project repo:
  [`ABLATION.md`](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/blob/main/ABLATION.md), [`TRAINING.md`](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/blob/main/TRAINING.md).
- FLEURS is human-read Wikipedia prose, so it contributes almost no conversational
  closers — 3,740 utterances yielded one `terima kasih`. Register matters more than size.

## Files beyond the configs

```
lexicon_raw/            vendored upstream lexicons, unmodified (MIT / Apache-2.0)
  granary/<lang>.txt    NVIDIA NeMo per-language phrase lists, 23 languages
external/               HALAS (CC BY 4.0) — 3,611 Earnings-22 clips with human
                        hallucination / looping spans across 9 ASR models
licenses/               upstream licence texts + NOTICE.md
```

HALAS per-model "hallucination or looping" counts out of 3,611. Note that
**`large-v3-turbo` hallucinates more than `large-v3`**:

| model | flagged |
|---|---:|
| whisper-large-v2 | 1,581 |
| parakeet-tdt-v2 | 1,217 |
| canary-1b | 1,213 |
| canary-1b-flash | 1,099 |
| phi-4 | 1,096 |
| whisper-large-v3-turbo | 1,060 |
| granite | 1,002 |
| whisper-large-v3 | 858 |
| CrisperWhisper | 772 |

## Provenance and licensing

Mixed, all permitting redistribution with attribution. Per component:

| component | licence | attribution |
|---|---|---|
| `reduplication`, `silence` audio | CC BY 4.0 | this dataset (synthesised, no recorded human speech) |
| `music` | CC BY 4.0 | Free Music Archive (commercial subset); per-row `artist` / `source_url` |
| `nonspeech` | CC BY 4.0 | FSD50K (`Fhrozen/FSD50k`) |
| `speech_in_noise` | CC BY 4.0 | derived: `genuine` speech × FSD50K backgrounds |
| `librispeech_test_clean` | CC BY 4.0 | LibriSpeech (`openslr/librispeech_asr`) |
| `genuine` | CC BY 4.0 | per-row `author` / `source_url` |
| `genuine_isolated` | CC0 / CC BY 4.0 / CC BY-SA 4.0, **per row** | per-row `author` / `source_url`; Lingua Libre via Wikimedia Commons |
| `lexicon` (merged), `targets`, `malaysian_sources` | CC BY 4.0 | this dataset |
| `lexicon_raw/agh_*` | MIT | AGH Signal Processing Group |
| `lexicon_raw/hf_whisper_hallucinations_phrases.csv` | MIT | Sacha Arbonel |
| `lexicon_raw/granary/*` | Apache-2.0 | NVIDIA, NeMo-speech-data-processor |
| `external/halas_*` | CC BY 4.0 | HALAS authors (AGH DSP) |

The synthetic audio contains **no recorded human speech** — source-filter synthesis and
generated noise — so it carries no speaker-consent or PII obligations. The `genuine` and
`genuine_isolated` arms are redistributed from public corpora under their own licences;
see `licenses/NOTICE.md`.

## Content warning

The lexicon records what an ASR model wrongly generates on audio containing no speech. Of
40,891 phrases, roughly 586 contain profanity and 583 violence-related terms
(`oh shit`, `i m not sure if i can get the gun`); 8 are sexual; none contain URLs. These
are **model errors, not anything a person said**, and the upstream sources carry the same
warning. They are kept because removing them would misrepresent the failure mode.

## Citation

Dataset: `Scicom-intl/Whisper-Hallucination`. Code: https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination

Built on:

- Koenecke, Choi, Mei, Schellmann, Sloane. *Careless Whisper: Speech-to-Text Hallucination Harms.* FAccT 2024.
- Barański et al. *Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio.* ICASSP 2025. (`arXiv:2501.11378`)
- Wang et al. *Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down.* Interspeech 2025. (`arXiv:2505.12969`)
- *HALAS: A Human-Annotated Dataset of Hallucinations of Modern ASR Systems.*
- NVIDIA `NeMo-speech-data-processor`, Granary pipeline.
- Lingua Libre / Wikimedia Commons contributors.
