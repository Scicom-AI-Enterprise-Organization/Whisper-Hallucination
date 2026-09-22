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
- config_name: lexicon_synth
  data_files:
  - split: train
    path: data/lexicon_synth/train-*.parquet
  - split: test
    path: data/lexicon_synth/test-*.parquet
---

# Whisper Hallucination and Repetition Probes

> **This is a benchmark. Every evaluation config is `test` — do not fine-tune on it.**
> `lexicon_synth` is the exception: synthetic training material with its own `train`/`test`
> split, and not one of the eight benchmark arms.
>
> To build training data, exclude the items in
> [`benchmark/exclusions.json`](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/blob/main/benchmark/exclusions.json)
> (546 FMA tracks, 1,168 FSD50K ids, 2,620 LibriSpeech utterances, the Malay stems). The
> benchmark draws FSD50K **eval** and FMA **shards 0–1**, so training can use FSD50K **dev**
> and FMA **shards 2–12** with no overlap.

Audio probes and phrase lexicons for two Whisper failure modes:

- **hallucination** — text with no sound behind it (`Terima kasih.` over silence)
- **repetition** — one unit emitted far more times than it was spoken

**Code and build scripts:** [github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination)

## The problem

Every common Whisper hallucination is also a phrase people say. `terima kasih` is the most
common Malay hallucination and the most common thing a call-centre agent says. A text
blocklist deletes both.

| audio | model says | correct action |
|---|---|---|
| silence | `Terima kasih.` | drop |
| music / noise | `Terima kasih.` | drop |
| speech in noise | part real, part invented | the hard case |
| someone says it | `Terima kasih.` | **keep** |

Silence is not the only trigger. Calm-Whisper measured large-v3 producing text on 99.97% of
UrbanSound8K clips, none of which contain speech. Energy is not evidence for words.

## Baselines

Five checkpoints, 11,852 clips each, greedy decoding, no forced language, no temperature
fallback. 2026-09-16/17 on 2× H20.

![Trade-off](https://raw.githubusercontent.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/main/bench/tradeoff.png)

No checkpoint is both quiet on noise and accurate on speech. They sit on a line (r = +0.68).

| | large-v2 | large-v3 | turbo | malaysian-v2 | M-turbo-v3 |
|---|---:|---:|---:|---:|---:|
| hallucinates on `nonspeech` | 98.8% | 89.9% | 80.7% | 76.0% | **9.3%** |
| emits *anything* on `silence` | 100% | 100% | 100% | 64.3% | **16.7%** |
| runaway on `reduplication` | 0.5% | 2.9% | 5.8% | 14.7% | 14.5% |
| emits nothing on `reduplication` | 0.0% | 0.1% | 5.2% | 16.9% | **58.0%** |
| recovers a phrase that IS spoken | 63.6% | **69.8%** | 68.0% | 31.3% | 41.1% |
| `librispeech` WER | 4.5% | 3.5% | 3.5% | 3.5% | 10.6% |

Ranked on hallucination alone, `Malaysian-turbo-v3` wins by a distance. Ranked on recovering
real speech it comes last. Measuring both on the same clips is the point.

Full tables: [README](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination#results).
Raw numbers: `bench/scores.json`. Harness: [`bench/`](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/tree/main/bench).

## Configs

| config | rows | hours | content | ground truth |
|---|---:|---:|---|---|
| `reduplication` | 1,440 | 1.9 | repeated units — syllables, vowels, laughter, clicks | exactly `n_repeats` repeats |
| `silence` | 42 | 0.4 | 6 noise floors × 7 durations | empty string |
| `music` | 600 | 4.5 | real Free Music Archive excerpts | empty string |
| `nonspeech` | 1,168 | 2.8 | FSD50K, label-verified voice-free | empty string |
| `speech_in_noise` | 1,200 | 3.1 | genuine speech + background, 5 SNRs | the real transcript |
| `genuine` | 4,694 | 12.4 | real Malaysian speech | human transcript |
| `genuine_isolated` | 88 | 0.05 | one phrase spoken alone | the phrase |
| `librispeech_test_clean` | 2,620 | 5.4 | English WER guard | human transcript |
| **`lexicon_synth`** | **29,112** | **15.8** | synthetic positives, 83 languages | the phrase |
| `lexicon` | 40,891 | — | known hallucination phrases, 100 languages | — |
| `ban_candidates` | 40,891 | — | each phrase classified safe/unsafe to blocklist | — |
| `targets` | 463 | — | phrases both hallucinated and genuinely said | — |
| `malaysian_sources` | 24 | — | survey of public Malaysian speech datasets | — |

Audio is 16 kHz mono FLAC, peak-normalised to −3 dBFS.

### Loading

`datasets >= 5.0` demands `torchcodec` to touch an audio column. Skip it:

```python
import io, soundfile as sf
from datasets import load_dataset, Audio

ds = load_dataset("Scicom-intl/Whisper-Hallucination", "reduplication", split="test")
ds = ds.cast_column("audio", Audio(decode=False))      # hand back raw bytes
x, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]), dtype="float32")
```

`streaming=True` works the same way. Lookup tables (`lexicon`, `ban_candidates`, `targets`,
`malaysian_sources`) load with `split="train"`.

## `lexicon_synth` — synthetic positives

Not a benchmark arm. Training material: 29,112 clips of lexicon phrases being spoken.

| | train | test |
|---|---:|---:|
| clips | 22,845 | 6,267 |
| hours | 12.49 | 3.32 |
| distinct phrases | 11,018 | 3,054 |
| languages | 75 | 83 |
| non-English share | 92% | 91% |
| named voices | 45 | 45 |

**Split by phrase, not by clip.** Each phrase is rendered in several voices. Splitting clips
would put `terima kasih` in both halves. Assignment is a hash of (phrase, language) with a
fixed salt.

**306 phrases are forced into `test`** because they appear in `targets` and drive the
`genuine_isolated` arm. The build verifies zero phrase overlap and zero benchmark phrases in
train, and refuses to write a release otherwise.

Phrases come from three places; `meta.phrase_provenance` says which:

| provenance | what it is |
|---|---|
| `observed` | phrases Whisper was seen to hallucinate (upstream lexicons) |
| `mined` | what these five checkpoints emitted on our non-speech arms |
| `translated` | English hallucination phrases rendered into 80 languages by NLLB-200 |

`translated` rows are **positives, not evidence**. A Tamil `thank you` is something people
say, not something Whisper was observed to invent in Tamil. Do not feed them back into
`ban_candidates`.

Audio comes from `Scicom-intl/Multilingual-Expressive-TTS-1.7B` (speaker-name conditioned)
and `k2-fsa/OmniVoice`, routed per language by measured CER. `meta` is a JSON string per clip
with model, codec, conditioning mode, speaker, sample rates, decoding parameters and seed.

### How many voices it really has

`voice` is a label, not a measurement.

![Voice diversity](https://raw.githubusercontent.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/main/tts/voice_diversity.png)

OmniVoice takes no speaker argument, so its clips carry `voice=""`. Measured with WavLM-sv
x-vectors on a calibrated scale (0.61 = different speakers, 0.85 = same speaker, both at
1.6 s), **33 of its 74 languages sat at 0.75 or worse** — one voice per language. The
`Multilingual-Expressive` clips measured 0.63, with same-name pairs at 0.83 and different-name
pairs at 0.60.

So 26 languages were **topped up, not replaced**: three extra named voices per phrase, the
auto-mode clip kept. 12,190 of 15,675 new clips passed the filter (77.8%), and the median
collapsed language moved 0.81 → 0.63. `meta.conditioning` distinguishes `speaker_name` from
`language_id`.

Seven languages were skipped (`am ml my sd si te sr`): the ASR judge cannot read them well
enough to validate any voice. That is a judge-coverage limit, not a quality claim.

### Filtering

Every clip passed an ASR round-trip: Whisper must recover the phrase, and the clip must be
about as long as the phrase should take. The gate is **per language, anchored to the judge's
own floor** — Whisper's CER on real FLEURS speech in 44 languages. A language where the judge
itself scores 0.88 (Burmese) or 1.23 (Amharic) cannot be held to a 0.25 gate. Nine such
languages are excluded rather than scored.

## Negative arms

### `reduplication`

A recording of `tu` repeated four times decoded as `tu` repeated until the token limit.
Inside a repeated-token region the decoder has no signal for how many repeats remain.

| field | values |
|---|---|
| `pattern` | `cv` (720), `vowel` (360), `laugh` (216), `click` (144) |
| `unit` | `tu ta ka pa da la na bi ko me` / `a i u e o` / `ha he hi` / `tick beep` |
| `n_repeats` | 3, 4, 5, 6, 8, 12 |
| `rate_hz` | 3, 5, 7, 9 |
| `tail_silence_s` | 0, 2, 8 |

`click` contains no speech, which separates repetition from speech as the trigger. Nothing
here is tied to a language.

Score with `n_hyp / n_repeats`: `1.0` correct, `>1` runaway, `<1` a mitigation that deleted
genuine repetition.

![What triggers the runaway](https://raw.githubusercontent.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/main/bench/reduplication_profile.png)

Laughter is the trigger (fine-tunes run away on 51–56% of laugh clips, 0% of clicks). More
repeats means more runaway. Trailing silence does not cause it.

### `silence`

Reference is the empty string, so any output is a hallucination. `floor` covers
`digital_zero`, `dither`, `hiss_-60db`, `hiss_-45db`, `hum_50hz`, `roomtone`. Durations
straddle the 30 s window: 1, 5, 10, 29, 31, 60, 120 s.

### `music`

Real produced music, not sound effects: Free Music Archive tracks (commercial subset,
CC BY 4.0), excerpted at 10 / 30 / 45 s so intros are skipped. This is what plays under a
call — hold music, a backing bed.

FMA ships no vocal tag, so some tracks contain singing and `vocals` records `unknown`. If you
need provably zero words, use `nonspeech`.

### `nonspeech`

FSD50K clips with no voice label of any kind — 583 music, 585 environmental and mechanical
noise. Anything labelled `Speech`, `Singing`, `Human_voice`, `Chatter`, `Laughter` or 20-odd
relatives was excluded, so the empty reference is verified by annotation.

### `speech_in_noise`

Genuine speech with a background bed, where transcription degrades into fabrication as
evidence weakens. 240 clips from `genuine` mixed with voice-free FSD50K at SNR +20, +10, +5,
0, −5 dB. Reference is the real transcript. 295 of 1,200 carry a target phrase.

## Positive arms

Without these, "hallucination rate fell" is unfalsifiable — a decoder that outputs nothing
scores perfectly on the negative arms.

### `genuine`

| source | clips | licence | register |
|---|---:|---|---|
| `SaLTUNIMAS/sarawak-malay-asr` | 1,164 | CC BY 4.0 | Sarawak Malay interviews |
| `emhaihsan/Synth-Manglish` | 2,457 | CC BY 4.0 | Manglish code-switching (**synthetic TTS**) |
| `google/fleurs` `ms_my` | 1,073 | CC BY 4.0 | human-read Wikipedia prose |

59 clips carry a high-risk phrase (`has_target_phrase = true`): `sama-sama` ×41,
`terima kasih` ×16, `selamat tinggal` ×2, `terima kasih banyak banyak` ×1. The rest are the
regression guard.

### `librispeech_test_clean`

LibriSpeech test-clean (CC BY 4.0), the split the mitigation literature reports its cost on.

Hallucination-space projection on large-v3 (arXiv:2609.04561):

| variant | ESC-50 HR | LibriSpeech WER clean / other |
|---|---:|---:|
| baseline | 44.25% | 4.06% / 5.87% |
| gated projection | 8.38% | 6.17% / 6.57% |
| always-on projection | 1.50% | **12.95% / 13.13%** |

Suppressing hallucination triples clean WER. That exchange rate is why the positive arms
exist.

### `genuine_isolated`

Native speakers from Lingua Libre saying one phrase and nothing else, across ms / id / en /
zh / ta. 88 clips. A hallucination surfaces as a bare `Terima kasih.`, and so does one of
these. In `genuine` the phrase sits mid-sentence, which is easier.

Licences vary per speaker (CC0 ×43, CC BY-SA 4.0 ×42, CC BY 4.0 ×3), so every row carries its
own `license`, `author` and `source_url`. Filter `license != "CC BY-SA 4.0"` to avoid
copyleft.

## `ban_candidates`

Every lexicon phrase crossed against all 7,402 genuine transcripts here (17.8 h):

| verdict | phrases | meaning |
|---|---:|---|
| `safe_candidate` | 871 | never in genuine speech, multi-word, frequently hallucinated |
| `unsafe` | 389 | occurs in genuine speech — banning it destroys real transcripts |
| `weak_evidence` | 39,631 | never observed, but too rare or short to be confident |

| phrase | verdict | hallucinated | genuine exact / substring |
|---|---|---:|---|
| `thanks for watching` | **safe_candidate** | 25,053 | 0 / 0 |
| `thank you` | **unsafe** | 31,353 | 0 / 13 |
| `terima kasih` | **unsafe** | 1 | **3** / **19** |

Use it in exact mode — drop an output only when the whole transcript equals the phrase.

**Caveat:** every ms / zh / ta phrase lands in `weak_evidence`, because the hallucination
counts come from public lexicons that barely cover those languages (6 Malay phrases total).
The classifier works — English demonstrates it — but the Malaysian side needs counts measured
locally.

## `lexicon`

Four public sources merged, normalised (NFC, casefold, punctuation-stripped), deduplicated on
`(phrase, lang)`:

| source key | rows | origin | licence |
|---|---:|---|---|
| `agh_boh` | 294 | AGH DSP "Bag of Hallucinations", ICASSP 2025 | MIT |
| `agh_full` | 30,407 | same paper, full non-speech tally | MIT |
| `hf_noise` | 7,889 | [`sachaarbonel/whisper-hallucinations`](https://huggingface.co/datasets/sachaarbonel/whisper-hallucinations) | MIT |
| `granary` | 3,028 | NVIDIA NeMo SDP, Granary pipeline | Apache-2.0 |

Top English entries: `thank you` (31,353), `thanks for watching` (25,053),
`thank you for watching` (6,264).

**Coverage is uneven.** English has 30,443 phrases. Malay has **6**. That gap is why this
dataset exists.

## Comparing hallucination rates

Published rates for the same model on the same corpus differ by more than 10×. Calm-Whisper
reports **99.97%** on UrbanSound8K for large-v3; arXiv:2609.04561 reports **76.08%**. The
difference is methodological: the latter counts only output surviving Whisper's own no-speech
filter (`no_speech_threshold = 0.6`). Always state which you mean.

Corpus composition matters too. arXiv:2609.04561 filtered FSD50K to clips labelled neither
speech, vocal nor music (HR 21.35%). Our `nonspeech` keeps music — filter `kind == "noise"` to
approximate their setup.

## Known limitations

- The positive arms are small relative to the negative ones; the phrase-matched subset is 147
  clips. Larger Malaysian conversational corpora are CC BY-NC and were excluded so this stays
  shareable.
- `Synth-Manglish` is TTS, not human speech, and is over half of `genuine`. Filter on `source`.
- `genuine` clips are long (median 9.6 s) while hallucinations happen on short audio.
  `genuine_isolated` is the short counterpart.
- `music` may contain singing. `nonspeech` is the label-verified alternative.
- `speech_in_noise` is synthetic mixing. SNR is exact, but channel effects and Lombard speech
  are absent.
- `lexicon_synth` is synthetic audio. It is training material, not evidence about what models
  hallucinate.
- No ablation results here. See
  [`ABLATION.md`](https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination/blob/main/ABLATION.md).

## Files beyond the configs

```
lexicon_raw/            vendored upstream lexicons, unmodified (MIT / Apache-2.0)
external/               HALAS (CC BY 4.0) — 3,611 Earnings-22 clips, human-annotated
                        hallucination / looping spans across 9 ASR models
licenses/               upstream licence texts + NOTICE.md
```

HALAS flagged counts out of 3,611. `large-v3-turbo` hallucinates more than `large-v3`:

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

All components permit redistribution with attribution.

| component | licence | attribution |
|---|---|---|
| `reduplication`, `silence` | CC BY 4.0 | this dataset (synthesised, no recorded human speech) |
| `music` | CC BY 4.0 | Free Music Archive; per-row `artist` / `source_url` |
| `nonspeech` | CC BY 4.0 | FSD50K (`Fhrozen/FSD50k`) |
| `speech_in_noise` | CC BY 4.0 | derived: `genuine` × FSD50K backgrounds |
| `librispeech_test_clean` | CC BY 4.0 | LibriSpeech (`openslr/librispeech_asr`) |
| `genuine` | CC BY 4.0 | per-row `author` / `source_url` |
| `genuine_isolated` | CC0 / CC BY 4.0 / CC BY-SA 4.0, per row | per-row `author` / `source_url` |
| `lexicon_synth` | CC BY 4.0 | synthesised by Multilingual-Expressive-TTS-1.7B and OmniVoice |
| `lexicon`, `targets`, `malaysian_sources` | CC BY 4.0 | this dataset |
| `lexicon_raw/agh_*` | MIT | AGH Signal Processing Group |
| `lexicon_raw/hf_whisper_hallucinations_phrases.csv` | MIT | Sacha Arbonel |
| `lexicon_raw/granary/*` | Apache-2.0 | NVIDIA, NeMo-speech-data-processor |
| `external/halas_*` | CC BY 4.0 | HALAS authors (AGH DSP) |

The synthetic audio contains no recorded human speech, so it carries no speaker-consent or
PII obligations. `genuine` and `genuine_isolated` are redistributed under their own licences;
see `licenses/NOTICE.md`.

Note on generators: OmniVoice's code is Apache-2.0 but its released weights are CC-BY-NC.
`Multilingual-Expressive-TTS-1.7B` is ours.

## Content warning

The lexicon records what an ASR model wrongly generates on audio containing no speech. Of
40,891 phrases, roughly 586 contain profanity and 583 violence-related terms; 8 are sexual.
These are **model errors, not anything a person said**. They are kept because removing them
would misrepresent the failure mode.

## Citation

Dataset: `Scicom-intl/Whisper-Hallucination`. Code:
https://github.com/Scicom-AI-Enterprise-Organization/Whisper-Hallucination

Built on:

- Koenecke, Choi, Mei, Schellmann, Sloane. *Careless Whisper: Speech-to-Text Hallucination Harms.* FAccT 2024.
- Barański et al. *Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio.* ICASSP 2025. (`arXiv:2501.11378`)
- Wang et al. *Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down.* Interspeech 2025. (`arXiv:2505.12969`)
- *HALAS: A Human-Annotated Dataset of Hallucinations of Modern ASR Systems.*
- NVIDIA `NeMo-speech-data-processor`, Granary pipeline.
- Lingua Libre / Wikimedia Commons contributors.
