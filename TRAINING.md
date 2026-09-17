# Core fix: fine-tuning whisper-large-v3

## Why weights, not decoding

Production serves on vLLM, which does not implement Whisper's transcription loop — no
`no_speech_threshold`, no `compression_ratio_threshold`, no `condition_on_previous_text`,
no temperature fallback, no `no_repeat_ngram_size`. Every "on top" fix either doesn't exist
there or lives outside the model (VAD, post-filter).

A fine-tuned checkpoint has none of those problems: it serves on vLLM exactly like the base
model. Weights are the only place a fix is both effective and deployable on this stack.

## The recipe to reproduce first

**Calm-Whisper** (Wang et al., Interspeech 2025) is the strongest published core fix and it
is remarkably cheap.

1. **Find the guilty heads.** Mask each of the 20 decoder *self-attention* heads in turn and
   measure hallucination rate on non-speech. They found masking head **#1** alone cuts it
   >30%, and heads **#1, #6, #11** account for **over 75%** of hallucinations.
2. **Calm-down fine-tune.** Update *only* those three heads across all decoder layers.
   Everything else frozen — that freeze is the anti-forgetting mechanism, which is why they
   need no positive data in the mix.
3. **Data:** non-speech audio paired with **blank labels**. 105 h (AudioSet, DEMAND, MUSAN).
4. **Hyperparameters:** batch 128, lr 1e-6, ~15% warmup, **5 epochs**.

Reported on large-v3: hallucination **99.97% → 15.51%**, LibriSpeech WER 2.12/4.07 →
2.19/4.13 — **under 0.1 pp degradation**.

Masking alone, with no training at all, already gets to 24.10%. That is the cheapest
possible first experiment and it should be run before any training.

## The LoRA trap

If LoRA is used instead of head-targeting: **α/r ≤ 0.2, not the usual α = 2r.** The common
heuristic causes catastrophic forgetting of end-of-sequence behaviour in Whisper's decoder —
and forgetting when to stop *is* the hallucination. Reported: dropping α/r from 2.0 to 0.2
eliminated hallucinations and improved WER. Knowledge distillation (α = 0.7) preserving the
teacher's non-speech token behaviour is the other reported route.

## Corpus status

`scripts/build_training_corpus.py` emits `train/val/test.jsonl`. Splits are disjoint by
source item — a track's excerpts, a clip's SNR variants, and a speaker never straddle a
boundary — and `librispeech_test_clean` is forced to test, because training on the WER guard
destroys the only honest number we have.

Current (eval-set scale, **not** training scale):

| pool | train hours | target |
|---|---:|---|
| blank-label non-speech | **7.1** | **105** |
| positive speech | 14.0 | — |
| reduplication | 1.7 | — |

**7% of the reference recipe.** Closing it needs roughly:

- FMA music, 11 unused shards → ~7,500 more tracks × 30 s ≈ **60 h**
- FSD50K dev, 35,676 voice-free clips ≈ **69 h** available
- synthetic silence variants — hours are free

Both are already wired (`build_music_arm.py`, `build_nonspeech_arm.py`). **This must run on
the box:** the laptop has ~15 GB free and the build needs ~6 GB of FLAC plus ~9 GB of source
parquet.

## Gates — a checkpoint ships only if all pass

| gate | arm | threshold |
|---|---|---|
| hallucination collapses | `silence`, `music`, `nonspeech` | HR ≤ 20% (reference: 15.51%) |
| English WER holds | `librispeech_test_clean` | ≤ +0.3 pp vs base |
| Malaysian WER holds | `genuine` (test split) | ≤ +0.3 pp vs base |
| model is not mute | `genuine`, `genuine_isolated` | empty-output rate ≤ base + 1 pp |
| loops bounded | `reduplication` | `overgen_p95` ≤ 1.5 |
| noise robustness holds | `speech_in_noise` | WER at 0 dB ≤ base + 1 pp |

The "not mute" gate matters most. Training on blank labels rewards silence, so the failure
mode of this recipe is a model that has learned to say nothing. `genuine_isolated` —
88 clips that are *nothing but* a short phrase — is the sharpest probe for it: those are
acoustically closest to the non-speech the model is being taught to ignore.

## Open question this repo can answer and the paper does not

Calm-Whisper targets non-speech hallucination only. **It says nothing about repetition.**
The `reduplication` arm (1,440 clips, exact known repeat counts) tests whether calming those
three heads also bounds runaway loops, or whether looping is a separate mechanism needing a
separate fix. Run it against base and calmed checkpoints — a cheap, novel result.

## Order of work

1. **Head-masking sweep, no training** — 20 heads × the non-speech arms. Reproduces the
   24.10% result and confirms the guilty heads on *our* data and languages.
2. **Scale the blank-label pool to ~105 h** on the box.
3. **Calm-down fine-tune** the identified heads; evaluate against every gate.
4. **Repetition probe** on the resulting checkpoint.
5. Only then consider LoRA or distillation variants.
