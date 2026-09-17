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

Three checkpoints × 8 arms × 11,852 clips, greedy decoding, fp16, 2× H20.
Full table in `DATASET_CARD.md`; raw numbers in `bench/scores.json`.

**Hallucination on non-speech** (reference is the empty string):

| arm | large-v2 | large-v3 | large-v3-turbo |
|---|---:|---:|---:|
| silence | 85.7% | 61.9% | 59.5% |
| music | 98.7% | 97.0% | 96.7% |
| nonspeech | 98.8% | 89.9% | 80.7% |

Counting a bare `"."` as output, it is **100.0% for every model on every non-speech arm**.

**Repetition runaway** (`reduplication`, 1,440 clips):

| | large-v2 | large-v3 | large-v3-turbo |
|---|---:|---:|---:|
| emitted/true repeats, p95 | 0.833 | 1.000 | **18.333** |
| longest token run | 440 | 440 | 440 |

`large-v3-turbo` over-generates repeats **18× at p95** while being the *best* of the three on
non-speech hallucination. `large-v2` under-counts instead. 440 is the token cap — the decoder
looped until it ran out of budget.

**WER guard:** LibriSpeech test-clean 4.5 / 3.5 / 3.5%. large-v3's 3.5% matches the published
figure, which is the check that the harness is wired correctly.

## Layout

```
bench/          benchmark harness: run_benchmark.py, score_benchmark.py, metrics.py, scores.json
scripts/        arm builders, lexicon/ban-list builders, HF release, corpus + disjointness tools
tts/            TTS candidate comparison (Scicom / OmniVoice / Higgs v2+v3) and synth harnesses
lexicon/        40,891 known hallucination phrases, 100 languages, 4 merged public sources
phrases/        targets.csv (high-risk phrases), ban_candidates.csv (safe/unsafe to blocklist)
benchmark/      exclusions.json — source items the benchmark burned; training must exclude them
sources/        survey of 24 public Malaysian speech datasets, with licences
manifests/      HALAS span-level human looping annotations (3,611 clips × 9 ASR models)
ablation/       configs.json — 34-config grid, vLLM-aware
audio/          generated + fetched arms (gitignored; rebuild from scripts/)
```

Docs: `CLAUDE.md` (gotchas — read first), `DATASET_CARD.md`, `SOURCES.md`, `ABLATION.md`,
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

Benchmark a checkpoint (**on the GPU box**, never the laptop):

```bash
uv venv --system-site-packages .venv && VIRTUAL_ENV=.venv \
  uv pip install transformers datasets accelerate soundfile

python bench/run_benchmark.py --model openai/whisper-large-v3 --device cuda:6
python bench/score_benchmark.py --lexicon lexicon/combined_lexicon.csv
```

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

Done: benchmark built, published, and baselined on three checkpoints. Training corpus staged
and verified disjoint. TTS generator selected by measurement.

Not done: the ablation itself (34 configs designed, none run), the core fine-tune, and the
100-language positive sweep. `CLAUDE.md` lists the benchmark's known weaknesses — read those
before quoting any number.
