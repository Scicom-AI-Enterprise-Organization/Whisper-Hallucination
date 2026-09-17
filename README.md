# Whisper-Hallucination

A benchmark, a training corpus, and an ablation plan for the two failure modes of
`whisper-large-v3`:

- **hallucination** — text with no phonetic basis in the audio (`Terima kasih.` over silence)
- **repetition / looping** — the decoder emits a unit far more times than it was spoken

**Dataset:** https://huggingface.co/datasets/Scicom-intl/Whisper-Hallucination

## The problem

Every high-frequency Whisper hallucination is also a phrase people genuinely say.
`terima kasih` is the most common Malay hallucination **and** the most common thing a
Malaysian call-centre agent says. A text blocklist cannot separate them, so every naive fix
deletes real transcriptions.

And silence is not the only trigger. Music and noise hallucinate at least as readily. The
real picture is a spectrum:

| state | audio | model says | correct action |
|---|---|---|---|
| silence | room tone | `Terima kasih.` | drop — invented |
| music / noise | a track, no voice | `Terima kasih.` | drop — energy, no evidence |
| speech in noise | real speech, low SNR | part real, part invented | the hard case |
| clean speech | someone says it | `Terima kasih.` | keep — correct |

So the dataset is **contrastive**: each high-risk phrase needs both a hallucinated and a
genuine pool, forcing a detector to decide from the *audio*.

## Measured baselines

Measured 2026-09-16/17 on 2× NVIDIA H20, `transformers` 5.17, fp16, **greedy decoding, no
forced language, no temperature fallback** — deliberately plain, so the numbers describe the
checkpoint rather than a decoding wrapper. Five checkpoints × 8 arms, all 11,852 clips per
model. Raw numbers in `bench/scores.json`.

Three OpenAI checkpoints and the two Malaysian fine-tunes
(`mesolitica/malaysian-whisper-large-v2`, `mesolitica/Malaysian-whisper-large-v3-turbo-v3`),
which turn out to sit on the opposite side of the trade-off this benchmark exists to expose.

![Benchmark results](bench/benchmark_results.png)

*Regenerate with `python bench/plot_benchmark.py` — it reads `bench/scores.json`.*

### Hallucination on non-speech — reference is the empty string

| arm | n | large-v2 | large-v3 | large-v3-turbo | malaysian-v2 | Malaysian-turbo-v3 |
|---|---:|---:|---:|---:|---:|---:|
| `silence` | 42 | 85.7% | 61.9% | 59.5% | 35.7% | **16.7%** |
| `music` | 600 | 98.7% | 97.0% | 96.7% | 69.7% | **31.0%** |
| `nonspeech` | 1,168 | 98.8% | 89.9% | 80.7% | 76.0% | **9.3%** |

The same rows counting **any output at all** — a bare `"."` included:

| arm | large-v2 | large-v3 | large-v3-turbo | malaysian-v2 | Malaysian-turbo-v3 |
|---|---:|---:|---:|---:|---:|
| `silence` | 100.0% | 100.0% | 100.0% | 64.3% | **16.7%** |
| `music` | 100.0% | 100.0% | 100.0% | 73.0% | **31.7%** |
| `nonspeech` | 100.0% | 100.0% | 100.0% | 77.8% | **9.4%** |

**Every OpenAI checkpoint emits something on every single non-speech clip.** The Malaysian
fine-tunes do not — `Malaysian-turbo-v3` returns a genuinely empty string on 90.6% of
voice-free FSD50K audio. Whatever else fine-tuning did, it taught the decoder that silence
is an acceptable answer, which is the one thing no amount of prompt or threshold tuning gets
out of the base checkpoints.

### Repetition runaway — `reduplication`, 1,440 clips

| metric | large-v2 | large-v3 | large-v3-turbo | malaysian-v2 | Malaysian-turbo-v3 |
|---|---:|---:|---:|---:|---:|
| emitted/true repeats, p50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| emitted/true repeats, **p95** | 0.833 | 1.000 | 18.333 | **55.000** | **73.333** |
| emitted/true repeats, max | 81.9 | 136.5 | 585.7 | 351.4 | 240.8 |
| clips over-generating >1.5× | 0.5% | 2.9% | 5.8% | **14.7%** | **14.5%** |
| clips emitting *nothing* | 0.0% | 0.1% | 5.2% | 16.9% | **58.0%** |
| longest consecutive token run | 440 | 440 | 440 | 440 | 440 |

That is the bill, and it comes twice. Both Malaysian fine-tunes run away on ~14.5% of clips
against 0.5–5.8% for the checkpoints they came from — a `max_run` of 440 is the
`max_new_tokens` cap, i.e. one token repeated until the budget ran out. And the same bias
that keeps `Malaysian-turbo-v3` quiet on noise makes it emit **nothing at all on 58% of
clips that do contain repeated speech**. On non-speech that silence is the right answer; on
the reduplication arm it is a deletion. The empty-output habit is not free.

### Word error rate — what a mitigation must not break

| arm | n | large-v2 | large-v3 | large-v3-turbo | malaysian-v2 | Malaysian-turbo-v3 |
|---|---:|---:|---:|---:|---:|---:|
| `librispeech_test_clean` | 2,620 | 4.5% | 3.5% | 3.5% | 3.5% | 10.6% |
| `genuine` | 4,694 | 35.4% | 31.8% | 31.8% | 42.3% | 60.7% |
| `genuine_isolated` | 88 | 23.9% | 23.9% | 29.5% | 42.0% | 46.6% |
| `speech_in_noise` | 1,200 | 41.6% | 34.5% | 36.1% | 51.1% | 58.7% |

`Malaysian-turbo-v3` looks catastrophic here, and the reading is wrong. Excluding only the
clips containing a token run ≥ 6 — **about 1% of them** — gives:

| arm | looped | WER | WER excluding looped clips |
|---|---:|---:|---:|
| `genuine` | 0.9% | 60.7% | **33.9%** |
| `speech_in_noise` | 1.0% | 58.7% | **25.6%** |
| `librispeech_test_clean` | 0.1% | 10.6% | **6.9%** |

![WER decomposition](bench/wer_decomposition.png)

**1% of clips carry 27 WER points**, because a looped clip emits tokens until the cap and
scores a WER in the tens. Set those aside and `Malaysian-turbo-v3` is the *best* model here
on noisy speech (25.6% against 32.7% for base turbo) and competitive on `genuine`. Its
failure is not transcription quality; it is that a small fraction of utterances run away.
`malaysian-v2` is the opposite case — its WER barely moves when looped clips are removed
(42.3% → 41.2%), so its regression is genuine quality loss, not runaway.

This is the whole argument for measuring both failure modes on the same clips. Ranked on
hallucination alone, `Malaysian-turbo-v3` wins by a distance. Ranked on raw WER it looks
unusable. Neither number alone describes it.

### What the models actually say

| model | on `silence` | on `music` |
|---|---|---|
| large-v2 | `you` ×18 · `. .` ×4 · `Yn ystod y 20. mlynedd, ma` ×4 | `so` ×49 · `បានានានានានានានានានានានានា` ×23 · `you` ×18 |
| large-v3 | `.` ×16 · `you` ×15 · `Thank you.` ×3 | `Thank you.` ×173 · `¶¶` ×86 · `.` ×16 |
| large-v3-turbo | `.` ×16 · `Thank you.` ×16 · `you` ×7 | `Thank you.` ×149 · `so` ×38 · `The End` ×31 |

These match the shipped `lexicon`: `Thank you.` is its top English entry (31,353 observations),
and `© BF-WATCH TV 2021` appears 13× on music. Language drift is visible too — large-v2 emits
Welsh on silence and a Khmer repetition loop on music.

### Reproducing

Inference runs **on the GPU box, never the laptop** (see `CLAUDE.md`), GPUs 6–7 only:

```bash
uv venv --system-site-packages .venv && VIRTUAL_ENV=.venv \
  uv pip install transformers datasets accelerate soundfile

python bench/run_benchmark.py --model openai/whisper-large-v3 --device cuda:6
python bench/run_benchmark.py --model mesolitica/Malaysian-whisper-large-v3-turbo-v3 --device cuda:7
python bench/score_benchmark.py --results bench/results --lexicon lexicon/combined_lexicon.csv
```

`bench/run_benchmark.py` (inference), `bench/score_benchmark.py` (metrics), `bench/metrics.py`
(metric definitions), `bench/scores.json` (these numbers).

## Generating positives: TTS and voice conversion

The benchmark's weakest arm is its positives — 52% of `genuine` is synthetic Manglish in a
single TTS voice. Fixing that needs both a multilingual generator and a way to put the same
phrase in many voices, so both were chosen by measurement rather than by reputation.

### TTS candidates

48 lexicon phrases, 22 languages, identical text, ASR round-trip CER (`tts/score_tts.py`;
summary re-derived by `tts/aggregate_tts.py`, which excludes the degenerate lexicon entries —
leaving them in puts `bn` at CER 22.9 and swamps every mean):

![TTS selection](tts/tts_results.png)

| | mean CER | median | wins | licence | coverage |
|---|---:|---:|---:|---|---|
| **Multilingual-Expressive-TTS-1.7B** | **0.221** | **0.000** | **18** | ours | untagged |
| OmniVoice | 0.349 | 0.018 | 3 | **Apache-2.0** | **97/100 langs** |
| Higgs Audio v2 | 1.089 | 0.713 | 0 | non-commercial | — |
| Higgs Audio v3 | 1.230 | 0.483 | 1 | non-commercial | 82/100 |

**OmniVoice** drives the 100-language sweep — its coverage is enumerable and the licence is
clean — and **Multilingual-Expressive** handles ms/en/zh/ta, where it wins outright. Higgs is
out on both counts; its licence independently forbids using outputs to train non-Boson speech
models.

### Voice conversion and cloning candidates

46 scorable phrases × 4 speaker slots = 184 clips per system (`tts/score_vc.py`, results in
`tts/vc_scores_summary.json`):

![Voice conversion results](tts/vc_results.png)

| | n | langs | CER | ΔCER | → target | ← source | gap | dur× | >2× | licence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Multilingual-Expressive** *(speaker name)* | 184 | **21** | **0.365** | **+0.022** | n/a | 11% | — | 1.00 | 4% | ours |
| kNN-VC | 184 | 20 | 0.441 | +0.099 | 68% | 59% | +9 | 0.99 | 0% | MIT |
| OpenVoice v2 (20 s ref) | 184 | 19 | 0.457 | +0.115 | 62% | 56% | +6 | 0.99 | 0% | MIT |
| OpenVoice v2 (6 s ref) | 184 | 20 | 0.467 | +0.125 | 61% | 53% | +8 | 0.99 | 0% | MIT |
| seed-vc (6 s ref) | 184 | 20 | 0.472 | +0.129 | 68% | 49% | +19 | 0.99 | 0% | GPL-3.0 |
| seed-vc (20 s ref) | 184 | **21** | 0.547 | +0.205 | 79% | 49% | +30 | 0.99 | 0% | GPL-3.0 |
| Higgs Audio v3 *(cloning)* | 184 | 20 | 0.627 | +0.285 | 84% | 12% | +72 | 0.90 | 3% | non-commercial |
| **Multilingual-Expressive** *(reference cloning)* | 184 | **21** | 1.740 | +1.397 | **103%** | 13% | **+90** | **3.27** | **68%** | ours |
| OpenVoice + MeloTTS *(cloning)* | 40 | **4** | 0.077 | −0.434 | 69% | 19% | +50 | 1.13 | 0% | MIT |
| CosyVoice 2 | **2** | — | — | — | — | — | — | — | — | Apache-2.0 |

**CER alone picks the wrong winner** — a converter that returns its input unchanged scores a
perfect 0 degradation and is worthless. So the table reports ΔCER *against the source clip*,
speaker similarity in both directions (the **gap** between them is the conversion that
actually happened), and **dur×**, the median output length divided by the source's. Those
percentages are calibrated, not raw cosines: at 1.6 s WavLM-sv scores 0.605 between different
speakers and 0.853 between two clips of the same one (`tts/calibrate_vc_sim.py`), so 0% reads
as "a stranger" and 100% as "the target".

**Multilingual-Expressive has two conditioning paths and they land at opposite extremes.**

- *Speaker name* — a name token from the `ExpressiveSpeech` inventory — is the intelligibility
  winner outright: **ΔCER +0.022**, four times better than the best converter, at the correct
  duration and 11% source retention. It cannot aim at a named target, because a name is not a
  reference you can score against.
- *Reference cloning* — reference transcript plus NeuCodec tokens, per the
  [base model card](https://huggingface.co/Scicom-intl/Multilingual-TTS-1.7B-Base#voice-cloning)
  — produces the **best voice match measured**, 103% of the calibrated scale with a +90 gap.
  It also does not stop: median **3.27× the source length**, 68% of clips over 2×, which is
  what drags CER to 1.740. Sampling, greedy and terminal punctuation all over-generate, and
  `generation_config.json` already sets `<|im_end|>` as EOS, so this is the model's behaviour
  on two-word targets rather than a misconfiguration.

**Read the 103% with care.** Longer audio gives a better x-vector estimate, and the
calibration was measured at 1.6 s while these clips run past 5 s, so the cloning row's
similarity is flattered by the very over-generation that ruins its CER. The +90 gap is real —
it is clearly the target's voice and not the source's — but it is not directly comparable
with the systems that stop on time.

**kNN-VC and OpenVoice barely convert.** Gaps of +6 to +9 mean the output sits almost equally
close to the target and to the source. Their good ΔCER is partly explained by not changing
much. seed-vc at +19 (+30 with a 20 s reference) is the best of the permissively-licensed
converters, and reference length is a dial: +11 points of identity for 0.076 more CER.

**What to use:**

- **Populating the positive pool → Multilingual-Expressive, speaker-name mode.** +0.022 ΔCER
  at the right duration across 21 languages. When the requirement is "different, intelligible
  voices" rather than "this specific speaker", nothing else is close.
- **Hitting a named target → Multilingual-Expressive reference cloning is the best voice
  match, but needs duration control first** — trimming to the phrase, or a stop condition.
  Out of the box, **Higgs Audio v3** is the best targeted cloner that terminates properly
  (84%, 0.90×), though its licence forbids using outputs to train non-Boson speech models.
  Among permissive licences, **seed-vc (6 s ref)**.

**Nobody else reaches the ceiling**, and the floor/ceiling distributions overlap heavily at
1.6 s. **Cloned TTS is a coverage story**: MeloTTS covers 4 of 21 languages and CosyVoice's
frontend is zh/en/ja/ko. **CosyVoice 2 could not be measured at all** — its flow encoder dies
with SIGFPE, which no `except` can catch.

```bash
python tts/build_vc_targets.py --device cuda:6      # 4 target speakers, medoid-filtered
bash tts/setup/install_vc.sh seedvc                 # one venv per candidate
python tts/vc_seedvc.py --device cuda:6 --ref short     # conversion candidates
python tts/clone_higgs3.py --device cuda:7              # cloning candidates
python tts/clone_scicom.py --mode clone --device cuda:6 # Scicom, reference-audio cloning
python tts/clone_scicom.py --mode named --device cuda:6 # Scicom, speaker-name conditioning
python tts/score_vc.py --systems tts/vc_out/* --device cuda:6
python tts/calibrate_vc_sim.py --device cuda:6          # the % scale above
python tts/plot_vc.py                                   # the figure above
```

## Layout

```
bench/          benchmark harness: run_benchmark.py, score_benchmark.py, metrics.py, scores.json
scripts/        arm builders, lexicon/ban-list builders, HF release, corpus + disjointness tools
tts/            TTS + voice-conversion candidate comparison, synth/convert harnesses, scorers
vc_repos/       cloned VC repos + their checkpoints (box only, gitignored)
lexicon/        40,891 known hallucination phrases, 100 languages, 4 merged public sources
phrases/        targets.csv (high-risk phrases), ban_candidates.csv (safe/unsafe to blocklist)
benchmark/      exclusions.json — source items the benchmark burned; training must exclude them
sources/        survey of 24 public Malaysian speech datasets, with licences
manifests/      HALAS span-level human looping annotations (3,611 clips × 9 ASR models)
ablation/       configs.json — 34-config grid, vLLM-aware
audio/          generated + fetched arms (gitignored; rebuild from scripts/)
```

Docs: `CLAUDE.md` (gotchas — read first), `DATASET_CARD.md` (the published card), `SOURCES.md`, `ABLATION.md`,
`TRAINING.md`.

## Quickstart

```bash
# Rebuild the synthetic arms locally — no download, exact ground truth
python scripts/build_lexicon.py
python scripts/build_targets.py
python scripts/make_silence_arm.py
python scripts/make_reduplication_arm.py
python scripts/validate_reduplication.py      # QA gate: audio matches its labels

# Fetch the real-audio arms
python scripts/fetch_audio.py aphasia halas musan
```

To benchmark a checkpoint, see [Reproducing](#reproducing) above.

## Using the published benchmark

Every audio config is `split="test"`. `datasets >= 5` demands `torchcodec` to touch an audio
column; skip it:

```python
import io, soundfile as sf
from datasets import load_dataset, Audio

ds = load_dataset("Scicom-intl/Whisper-Hallucination", "reduplication", split="test")
ds = ds.cast_column("audio", Audio(decode=False))
row = ds[0]
x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")   # 16 kHz mono
```

| config | rows | content |
|---|---:|---|
| `silence` | 42 | 6 noise floors × 7 durations |
| `music` | 600 | real Free Music Archive excerpts |
| `nonspeech` | 1,168 | FSD50K, label-verified voice-free |
| `speech_in_noise` | 1,200 | genuine speech + background, 5 SNRs |
| `reduplication` | 1,440 | repeated units, exact known counts |
| `genuine` | 4,694 | real Malaysian speech |
| `genuine_isolated` | 88 | one phrase spoken alone |
| `librispeech_test_clean` | 2,620 | English WER guard |
| `lexicon` / `ban_candidates` / `targets` / `malaysian_sources` | — | lookup tables |

## Training corpus

Built separately and verified disjoint — see `CLAUDE.md`. Currently on the box:
**32,527 clips / 111 h**, of which **94.1 h blank-label** (90% of the Calm-Whisper recipe's
105 h), with **0 collisions** against the benchmark's 8,956 exclusion keys.

```bash
python scripts/freeze_benchmark.py          # emit exclusions.json
python scripts/build_training_corpus.py     # emit train/val jsonl (box)
python scripts/verify_disjoint.py           # must print 0 collisions
```

## Status

Done: benchmark built, published, and baselined on **five** checkpoints — three OpenAI plus
`mesolitica/malaysian-whisper-large-v2` and `mesolitica/Malaysian-whisper-large-v3-turbo-v3`.
Training corpus staged and verified disjoint. TTS generator and voice-conversion model both
selected by measurement (`CLAUDE.md`: *TTS selection* and *VC selection, as measured*).

Not done: the ablation itself (34 configs designed, none run), the core fine-tune, and the
100-language positive sweep. `CLAUDE.md` lists the benchmark's known weaknesses — read those
before quoting any number.
