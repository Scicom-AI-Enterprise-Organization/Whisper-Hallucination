# Ablation: reducing hallucination and repetition on whisper-large-v3

## Engine reality: production is vLLM

**Most of the original grid does not exist on the production stack.** vLLM serves Whisper
as a model; it does not implement openai-whisper's *transcription loop*. Unavailable:

| knob | vLLM |
|---|---|
| `no_speech_threshold` | ✗ — and `verbose_json` does not even return `no_speech_prob` |
| `compression_ratio_threshold` | ✗ |
| `logprob_threshold` | ✗ |
| `condition_on_previous_text` | ✗ |
| temperature-fallback ladder | ✗ (single temperature only) |
| `no_repeat_ngram_size` | ✗ |
| `hallucination_silence_threshold` | ✗ |
| hallucination-space projection (arXiv:2609.04561) | ✗ — needs decoder hidden states mid-forward |

What vLLM's `/v1/audio/transcriptions` *does* expose: `temperature`, `top_p`, `top_k`,
`min_p`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, and beam
search (documented as "highly inefficient" for encoder-decoder models).

**The cascade cannot happen here.** vLLM chunks audio into independent 30 s windows and
does not carry previous text across them (open request, vLLM PR #20249). So the
self-reinforcing loop — one hallucination becoming the next window's prompt — is
structurally impossible on this stack. `condition_on_previous_text=False`, which the
faster-whisper grid treats as the single largest lever, **is already in effect for free.**

What remains is *within-window* runaway: the `tutututu` case, where a loop consumes one
window's token budget. That is bounded directly by `max_tokens`, which makes it the
cheapest high-value fix available — subject to checking the cap never clips real speech
(`librispeech_test_clean`, `genuine`).

Expect the vLLM baseline to loop **less** than a faster-whisper baseline. That is an
architectural difference, not evidence that vLLM is a better model.

Two further consequences:

1. `hall_rate_filtered` cannot be computed from a vLLM response. It is still scored under
   faster-whisper, because it explains the 10x disagreement between published baselines,
   but it is a research number, not a production one.
2. **The leverage moves outside the engine.** Since silence, music and noise all trigger
   hallucination and none of the gating knobs exist here, the shippable levers are a VAD
   front-end, the chunking policy, `repetition_penalty` / `max_tokens`, and a post-hoc
   filter. Those are the `vllm` and `external` stages in `configs.json`.

The faster-whisper-only configs are kept and labelled, because they still diagnose *where*
a failure comes from — but a fix that only exists there cannot ship.

## Design

34 configs in `ablation/configs.json`, six stages:

1. **OFAT** (15) — one knob moved off `baseline` at a time, so each effect is attributable.
2. **Combo** (4) — the winners stacked, including a deliberately over-aggressive `combo_full`.
3. **Model** (3) — large-v3 vs turbo vs distil vs CrisperWhisper.
4. **Intervention** (2) — hallucination-space projection, gated and always-on
   (arXiv:2609.04561). Training-free, but needs transformers; **not runnable on vLLM**.
5. **vLLM** (6) — the knobs that actually exist on the production stack.
6. **External** (3) — VAD front-end, lexicon post-filter, and the two combined. Engine-
   independent, so these survive a stack change.

## The post-filter, measured

`phrases/ban_candidates.csv` crosses all 40,891 lexicon phrases against every genuine
transcript we hold (7,402 transcripts, 17.8 h) and classifies each one:

| verdict | phrases | meaning |
|---|---:|---|
| `safe_candidate` | 871 | never observed in genuine speech, multi-word, frequently hallucinated |
| `unsafe` | 389 | occurs in genuine speech — banning it destroys real transcripts |
| `weak_evidence` | 39,631 | never observed, but too rare or too short to be confident |

This settles the founding question empirically:

| phrase | verdict | hallucinated | genuine (exact / substring) |
|---|---|---:|---|
| `thanks for watching` | **safe_candidate** | 25,053 | 0 / 0 |
| `thank you` | **unsafe** | 31,353 | 0 / 13 |
| `terima kasih` | **unsafe** | 1 | **3** / **19** |

`terima kasih` is banned-by-instinct and would cost 3 whole-utterance and 19 in-context
real transcriptions in only 17.8 h of audio. `thanks for watching` is free to drop.

**The limitation is Malay evidence, not the method.** Every ms/zh/ta phrase lands in
`weak_evidence` because the *hallucination counts* come from public lexicons that barely
cover those languages (6 Malay phrases total). The classifier works — English proves it.
Filling in the Malaysian side needs our own probe run, which is what the GPU time is for.

**Engine note:** the `hsp_*` configs need HF transformers, not faster-whisper. CTranslate2
does not expose decoder hidden states, and the method edits them mid-forward (layer 28,
rank 4, α = 1.0 for large-v3; subspace from SVD over hallucinating-minus-empty activations
on ~1,200 non-speech calibration clips).

Every config is scored on **every arm**. That is the point of the design: the knobs that
suppress hallucination are the same knobs that delete genuine speech, so a config only
counts as a win if the non-speech arms improve **and** the genuine arms hold.

## The knobs, and what each one is actually attacking

| config | knob | mechanism it targets |
|---|---|---|
| `no_condition` | `condition_on_previous_text=False` | **the cascade.** Each 30 s window's output is fed back as the next window's prompt, so one hallucination reinforces itself. Expected largest single effect on long-form looping. |
| `vad_on` / `vad_strict` | Silero VAD front-end | the encoder never sees the silence. WhisperX's finding: external VAD boundaries beat decoded timestamp tokens. |
| `nospeech_low` | `no_speech_threshold 0.6→0.4` | more willing to call a window silent |
| `cr_strict` | `compression_ratio_threshold 2.4→1.8` | Whisper's built-in repetition detector; 2.4 lets short loops through |
| `reppen_*` | `repetition_penalty` | direct decoder pressure against repeats |
| `norepeat_*` | `no_repeat_ngram_size` | hard ban on repeating any n-gram |
| `greedy_only` | drop temperature fallback | fallback escapes loops but resampling at T=1.0 also invents text |
| `logprob_strict` | `log_prob_threshold -1.0→-0.5` | hallucinated spans are low-confidence |

`repetition_penalty` and `no_repeat_ngram_size` are faster-whisper (CTranslate2) only.
openai-whisper instead exposes `hallucination_silence_threshold`.

## The trade-off this is built to expose

`norepeat_3` forbids repeating any trigram. It should crush looping — and it should also
make `tu tu tu tu` **untranscribable**, because that utterance *is* a repeated trigram.
The reduplication arm exists to put a number on that cost. Expect:

- non-speech arms: `hall_rate` ↓ a lot
- reduplication arm: `overgen_p95` ↓ toward 1.0, but `overgen_p50` drops **below** 1.0 —
  the model now under-counts genuine repeats

A config that only reports the first line is mis-sold. `combo_conservative`
(mechanism fixes only, nothing that forbids repetition) is the intended production candidate.

## Metrics

| metric | arm | reading |
|---|---|---|
| `hall_rate` | silence, musan, esc50 | share with any output. Ground truth is `""`, so lower is strictly better. |
| `loop_rate`, `max_run` | all | longest **consecutive** n-gram run ≥ threshold |
| `overgen_p50` / `p95` | reduplication | predicted ÷ true repeats. 1.0 correct, >1 runaway, <1 over-suppressed. |
| `lexicon_rate` | all | output is exactly a known hallucination phrase (40,891-phrase lexicon) |
| `wer`, `cer` | fleurs, halas | **regression guard** |

## Priors from the literature — and why they do not agree

Published baseline hallucination rates for the **same model on the same corpus** differ by
more than an order of magnitude:

| source | corpus | large-v3 baseline HR |
|---|---|---:|
| Calm-Whisper (Interspeech 2025) | UrbanSound8K | **99.97%** |
| Hallucination Space Projection (arXiv:2609.04561) | UrbanSound8K | **76.08%** |
| Hallucination Space Projection | ESC-50 | 44.25% |
| Hallucination Space Projection | FSD50K | 21.35% |

The gap is **methodological, not empirical**. arXiv:2609.04561 defines HR as *any non-empty
output remaining after no-speech filtering* — Whisper's own `no_speech_threshold = 0.6`
filter runs first. Calm-Whisper reports the raw rate. So they are measuring different
things, and neither number is transferable to our setup.

Two consequences for this ablation:

1. **Report HR both ways** — raw, and after the built-in no-speech filter. A config that
   only moves one of them has not done what it appears to.
2. **FSD50K is the easy end of the range** (21% vs 76%). arXiv:2609.04561 filtered out
   clips labelled speech, vocal **or music**; our `nonspeech` arm deliberately *keeps*
   music (583 music / 585 noise, `kind` column). Filter to `kind == "noise"` to approximate
   their setup. Our `music` arm (real produced tracks) should sit at the hard end.

Other anchors:

- Calm-Whisper: head masking → 24.10%, fine-tuning → 15.51%.
- SAE steering: 86.88% → 27.33% on large-v3 without touching weights.
- HALAS: large-v3 flagged on 858/3,611 real earnings-call clips; turbo is **worse** (1,060).

## Correction: suppression is not free

An earlier draft of this plan said that reaching ~15% on non-speech "without moving WER"
would match fine-tuning at zero cost. **arXiv:2609.04561 shows WER does move**, and it
quantifies the trade-off on large-v3:

| variant | ESC-50 HR | LibriSpeech WER (clean / other) |
|---|---:|---:|
| baseline | 44.25% | 4.06% / 5.87% |
| gated projection | 8.38% | 6.17% / 6.57% |
| always-on projection | 1.50% | **12.95% / 13.13%** |

Always-on nearly eliminates hallucination and **triples clean WER**. Even the gated variant
costs ~2.1 pp. So the question is never "how low can HR go" but "what is the exchange rate",
which is exactly what scoring every config on every arm is for.

`librispeech_test_clean` (2,620 clips) exists in this repo so our WER numbers land on the
same axis as that table.

## The hypothesis our arms are positioned to test

arXiv:2609.04561 gates its intervention on **Whisper's own no-speech probability**
(γ = 0.05 for large-v3), and measures false rejection on *clean* LibriSpeech: 0.41–9.97%.

Background music raises `no_speech_prob` on audio that **does** contain speech. So the gate
should misfire exactly where it matters operationally — an agent talking over a hold-music
bed. `speech_in_noise` (1,200 clips, SNR +20 → −5 dB, music and noise backgrounds, real
transcripts) is built to measure that, and nothing in the published work covers it: their
cost measurement is clean English read speech only.

Prediction to falsify: **false rejection rises sharply below ~+5 dB SNR**, and more so for
music backgrounds than noise, because music is the condition the gate was tuned to fire on.

## Run order

faster-whisper cannot serve at production scale, so GPU time goes to what can ship:

1. **`baseline` + `vllm` (6) + `external` (3)** — the shippable set. Run these first.
2. **`ofat` / `combo` / `model` (22)** — diagnostic. They explain *where* a failure comes
   from; they cannot be deployed. Run only if time allows.
3. **`intervention` (2)** — needs a transformers deployment that does not exist here.

## Running it

On the GPU box only:

```bash
python scripts/run_ablation.py --all --arms silence reduplication aphasia_koenecke halas_earnings22
python scripts/score_ablation.py
```

Resumable — existing `ablation/results/<config>/<arm>.jsonl` is skipped.

## Scope beyond Malaysian

The arms are language-neutral by construction: `silence` has no language, and
`reduplication` sweeps CV syllables, bare vowels, laughter and clicks rather than words.
The lexicon already covers 100 languages. So the same grid extends past ms/en/zh/ta by
adding language-specific **positive** pools only — the negative side is already global.
