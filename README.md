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

The arm is a factorial grid — 4 unit patterns × 6 repeat counts × 4 rates × 3 tail
silences — so the corpus means above can be sliced by what the audio actually did:

![What triggers the runaway](bench/reduplication_profile.png)

*Regenerate with `python bench/reduplication_profile.py` on the box (per-clip results), then
`python bench/plot_reduplication.py`.*

**The runaway is driven by the speech, not the silence after it.** Laughter is the trigger:
the fine-tunes run away on 51–56% of laugh clips and on 0% of clicks, with base turbo at
29%. Every model runs away more the more times the unit is repeated (3 → 12 repeats takes the
fine-tunes from ~5% to ~23%). Trailing silence, the usual suspect, does *not* raise it — for
the base checkpoints 8 s of silence *lowers* runaway (turbo 8.5% → 2.7%). The deletion
habit has its own shape: `malaysian-v2` is silent on 96% of clicks (arguably right — a
click is not speech) and almost never on syllables or laughter, while `Malaysian-turbo-v3`
is silent on 46–75% of everything and increasingly so the faster the repeats come.

One caveat the slicing exposes: `run ≥ 6 tokens` is not a clean loop signal on *this* arm,
because a correct transcript of six repeats is itself a run of six — large-v3's loop rate
jumps from 1.7% at five repeats to 37.9% at six. Read `runaway >1.5×` and `emits nothing`
here; `run ≥ 6` is for the speech arms.

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

### The false-positive side — when the phrase is really spoken

The three non-speech arms measure what a model invents. They cannot tell you what a
mitigation would cost, because none of those clips contain speech. `lexicon_synth`'s `test`
split is the other half of that contrast: **6,267 clips, 83 languages, of the hallucination
phrases actually being said**. Emitting the phrase here is the correct answer, and every miss
is a real transcription a filter would delete.

Production audio is not a bare 0.9 s clip, so each clip is run three ways — and the middle
condition is a control, because **Whisper zero-pads every input to a 30 s window anyway**:
a 0.9 s clip already sits in ~29 s of digital zeros, so trailing zeros cannot change anything
and leading zeros only shift where the speech starts. Room tone is a different signal
entirely, and it is drawn from the `silence` arm itself.

![lexicon_synth results](bench/lexicon_synth_results.png)

*Regenerate with `python bench/run_benchmark.py --arms lexicon_synth [--pad-lead 2 --pad-tail 2
--pad-kind roomtone --pad-tag __tone2]`, then `python bench/score_lexicon_synth.py` and
`python bench/plot_lexicon_synth.py`.*

| model | recovered (bare → tone) | median CER (bare → tone) | mean CER (bare → tone) | >2× output | emits nothing |
|---|---|---|---|---|---|
| whisper-large-v2 | 63.6% → 63.4% | 0.125 → 0.125 | 0.378 → 0.384 | 0.6% → 0.7% | 0.0% |
| **whisper-large-v3** | **69.8% → 68.5%** | **0.089 → 0.091** | **0.288 → 0.302** | 0.2% → 0.4% | 0.0% |
| whisper-large-v3-turbo | 68.0% → 67.0% | 0.087 → 0.091 | 0.348 → 0.382 | 0.7% → 0.9% | 0.0% |
| malaysian-whisper-v2 | 31.3% → 29.6% | 0.800 → 0.826 | 0.770 → **1.151** | 2.6% → 3.7% | 0.0% |
| Malaysian-turbo-v3 | 41.1% → 35.4% | 0.385 → 0.529 | 1.508 → **3.749** | 3.0% → 5.7% | 2.8% → 1.1% |

**The ranking inverts.** On the negative arms `Malaysian-turbo-v3` is the best checkpoint by a
distance — it stays silent on 90.6% of voice-free audio where every OpenAI model emits
something on 100%. Here it recovers the phrase on **35–41%** of clips against large-v3's
**68–70%**. The quiet is not selective; it is a prior against emitting text, and it costs
real transcriptions. The same trade shows in `malaysian-whisper-v2` at 30%.

**Room tone is the expensive condition, and only for the fine-tunes.** The base checkpoints
barely move (large-v3 +0.014 mean CER, recovery −1.3 points). `malaysian-whisper-v2` gains
**+0.38 mean CER** and `Malaysian-turbo-v3` **+2.24** from nothing but a noise floor around
the same speech. The zeros control moves far less in every case, so this is the noise floor
rather than the position shift — and it is exactly the condition a VAD chunk delivers.

It costs throughput too, for the same reason: over the identical 6,267 clips, the base
checkpoints are flat under padding (±4%) while `malaysian-whisper-v2` goes 496 s → 627 s
(+26%) and `Malaysian-turbo-v3` 452 s → 516 s. More tokens, more time.

**The median clip is fine; a tail does the damage.** Median CER moves by hundredths while mean
CER triples, because 3–6% of clips run past 2× the phrase length. That is the same runaway
the `reduplication` arm isolates, reappearing on ordinary speech once silence surrounds it.

One caveat stated plainly: the corpus was filtered by `whisper-large-v3` with the language
forced, so v3 is scored partly on clips selected by its own agreement, and its lead over
large-v2 and turbo should be read with that in mind. The gap to the Malaysian fine-tunes is
far too large to be explained by it. Decoding here is auto-language, as everywhere in this
harness, which is why absolute recovery tops out near 70%.

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

**A third of the Multilingual-Expressive clips in that table used an unverified speaker
name.** The grid assigned one of three names per phrase, and one of them —
`multilingual-tts_audio_Rahman` — is not in `ExpressiveSpeech`, the same mistake that
invalidated the voice-conversion named-mode row. An unknown name does not raise; it
conditions on a token the fine-tune never saw. Split by name, over the 16 non-degenerate
phrases each:

| speaker | in inventory | mean CER | median |
|---|---|---:|---:|
| `multilingual-tts_audio_Grace` | yes | 0.130 | 0.000 |
| `DisfluencySpeech` | yes | 0.254 | 0.083 |
| `multilingual-tts_audio_Rahman` | **no** | 0.280 | 0.000 |

Dropping those phrases **for every system** (`python tts/aggregate_tts.py
--valid-speakers-only`) leaves 32 clips over 20 languages and does not change the ranking —
it widens it: Multilingual-Expressive **0.192** (17 wins), OmniVoice 0.450 (2), Higgs v2
1.158 (0), Higgs v3 1.583 (1). So the published **0.221 understates it**, and the table above
is kept as the headline because it is the full 22-language grid. Verify speaker names against
`tts/expressive_speakers.json` before trusting any named-mode number.

### Voice conversion and cloning candidates

46 scorable phrases × 4 speaker slots = 184 clips per system (`tts/score_vc.py`, results in
`tts/vc_scores_summary.json`).

**OmniVoice is the source, so it is the baseline every other row is measured against.**
Every converter is fed the same OmniVoice rendering of each phrase, so `ΔCER` is the
degradation a system adds *on top of that clip* and `← source` is similarity to the
OmniVoice voice it started from. Its own intelligibility is in the TTS table above.

That makes it the zero point, but **not** ineligible as a candidate: OmniVoice also takes a
reference clip (`generate(text=, ref_audio=, ref_text=)`), so it can aim at a target speaker
exactly like Higgs v3 does. `tts/clone_omnivoice.py` runs that arm, and it now has a row —
the interesting one, since OmniVoice is the generator the 100-language sweep runs on and the
question was whether conditioning on a voice costs it intelligibility. It does: **+0.339 CER
over its own auto-mode clip**, second worst in the table. What it buys is the strongest
identity transfer measured here (**107%**, past the same-speaker ceiling, gap **+92**) under
the only permissive licence in that bracket. (The one
exception is `OpenVoice + MeloTTS`, where MeloTTS synthesises the audio and OpenVoice
re-voices it, so its source is MeloTTS; that is also why its ΔCER is negative — MeloTTS is
simply more intelligible than OmniVoice in the four languages it covers.)

![Voice conversion results](tts/vc_results.png)

| | n | langs | CER | ΔCER | → target | ← source | gap | dur× | >2× | licence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Multilingual-Expressive** *(speaker name)* | 184 | **21** | **0.308** | **−0.035** | n/a | 13% | — | 1.03 | 3% | ours |
| kNN-VC | 184 | 20 | 0.437 | +0.095 | 68% | 59% | +9 | 0.99 | 0% | MIT |
| OpenVoice v2 (20 s ref) | 184 | 19 | 0.457 | +0.115 | 62% | 56% | +6 | 0.99 | 0% | MIT |
| OpenVoice v2 (6 s ref) | 184 | 20 | 0.466 | +0.124 | 61% | 53% | +8 | 0.99 | 0% | MIT |
| seed-vc (6 s ref) | 184 | 20 | 0.472 | +0.129 | 68% | 49% | +19 | 0.99 | 0% | GPL-3.0 |
| seed-vc (20 s ref) | 184 | **21** | 0.549 | +0.206 | 79% | 49% | +29 | 0.99 | 0% | GPL-3.0 |
| Higgs Audio v3 *(cloning)* | 184 | 20 | 0.653 | +0.311 | 84% | 12% | +72 | 0.90 | 3% | non-commercial |
| **OmniVoice** *(cloning)* | 183 | **21** | 0.683 | +0.339 | **107%** | 15% | **+92** | 1.28 | 1% | **Apache-2.0** |
| **Multilingual-Expressive** *(reference cloning)* | 184 | **21** | 1.740 | +1.397 | 103% | 13% | +90 | **3.27** | **68%** | ours |
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
  winner outright: **ΔCER −0.035** — it is *more* intelligible than the OmniVoice clip it was
  asked to re-voice, the only system to manage that across all 21 languages — at the correct
  duration (1.03×) and 13% source retention. It cannot aim at a named target, because a name
  is not a reference you can score against.

  That row was wrong until 2026-09-18: one of its four speakers,
  `multilingual-tts_audio_Rahman`, **is not in `ExpressiveSpeech`** (the only Rahman there is
  `genshin-voice_audio_Rahman`, 46 rows of Japanese). An unknown name does not raise — it
  conditions on a token the fine-tune never saw — so a quarter of the grid was effectively
  unconditioned and the row read ΔCER +0.022 / CER 0.365. With a real name in that slot
  (`multilingual-tts_audio_Ryan`) it is ΔCER −0.035 / CER 0.308. **Verify speaker names
  against the dataset before trusting a named-mode number.**
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
much. seed-vc at +19 (+29 with a 20 s reference) is the best of the permissively-licensed
*converters*, and reference length is a dial: +10 points of identity for 0.077 more CER.

**What to use:**

- **Populating the positive pool → Multilingual-Expressive, speaker-name mode.** −0.035 ΔCER
  at the right duration across 21 languages: it *improves* on the source clip rather than
  degrading it. When the requirement is "different, intelligible voices" rather than "this
  specific speaker", nothing else is close.
- **Hitting a named target → Multilingual-Expressive reference cloning is the best voice
  match, but needs duration control first** — trimming to the phrase, or a stop condition.
  Out of the box, **Higgs Audio v3** is the best targeted cloner that terminates properly
  (84%, 0.90×), though its licence forbids using outputs to train non-Boson speech models.
  Among permissive licences the choice is now a trade, not a compromise: **OmniVoice**
  (Apache-2.0) buys the strongest identity in the table — 107%, gap +92, and it very nearly
  terminates (1.28×, 1% over 2×) — for +0.339 ΔCER, while **seed-vc (6 s ref)** keeps
  intelligibility (+0.129) and moves the voice far less (+19). Pick by which side of that
  trade the positive pool needs.

**OmniVoice reaches the ceiling too** — 107%, the only other system past it, and with the
same caveat: it runs 1.28× long, so its x-vector gets more audio than the 1.6 s calibration
assumed. The floor/ceiling distributions also overlap heavily at 1.6 s. **Cloned TTS is a coverage story**: MeloTTS covers 4 of 21 languages and CosyVoice's
frontend is zh/en/ja/ko. **CosyVoice 2 could not be measured at all** — its flow encoder dies
with SIGFPE, which no `except` can catch.

```bash
python tts/build_vc_targets.py --device cuda:6      # 4 target speakers, medoid-filtered
bash tts/setup/install_vc.sh seedvc                 # one venv per candidate
python tts/vc_seedvc.py --device cuda:6 --ref short     # conversion candidates
python tts/clone_higgs3.py --device cuda:7              # cloning candidates
python tts/clone_omnivoice.py --device cuda:7           # OmniVoice, reference cloning
python tts/clone_scicom.py --mode clone --device cuda:6 # Scicom, reference-audio cloning
python tts/clone_scicom.py --mode named --device cuda:6 # Scicom, speaker-name conditioning
python tts/score_vc.py --systems tts/vc_out/* --device cuda:6   # ALL arms, one venv
python tts/make_vc_summary.py                           # summary the figure reads
python tts/calibrate_vc_sim.py --device cuda:6          # the % scale above
python tts/plot_vc.py                                   # the figure above
```

### Synthesising the positive pool — and how many voices it really has

The TTS and VC tables above pick the generators; `tts/build_lexicon_queue.py` →
`tts/synth_lexicon.py` → `tts/filter_lexicon_synth.py` runs them over the lexicon, and the
result ships as the `lexicon_synth` config: **29,112 clips, 15.8 hours, 83 languages**, split
`train` (22,845) / `test` (6,267) by phrase, with benchmark phrases forced into `test`.

**A `voice` column is a label, not a measurement.** OmniVoice takes no speaker argument at
all, so every clip it renders carries `voice=""` — and measuring them with the same WavLM-sv
judge and calibrated scale as the VC table (0.605 between different speakers, 0.853 between
two clips of one, both at 1.6 s) showed **33 of its 74 languages at a median pairwise cosine
of 0.75 or worse**: one voice per language. The `Multilingual-Expressive` half measured 0.63,
with same-name pairs at 0.826 and different-name pairs at 0.601 — genuinely distinct speakers,
and proof the name conditioning lands.

![Voice diversity](tts/voice_diversity.png)

*Regenerate with `python tts/voice_diversity.py` on the box, then `python
tts/plot_voice_diversity.py`.*

Two ways to fix it, probed on the same 4 collapsed languages × 24 phrases × 4 voices and
scored in one venv:

| route | yield mk / gu / it | median CER | different-voice cosine |
|---|---|---:|---|
| **Multilingual-Expressive, speaker name** | **86% / 80% / 79%** | 0.07 / 0.17 / 0.00 | **0.652 / 0.638 / 0.578** |
| OmniVoice, reference cloning | 46% / 68% / 53% | 0.32 / 0.25 / 0.26 | 0.684 / 0.722 / 0.687 |
| *auto mode, for reference* | 89% / 73% / 83% | 0.02 / 0.11 / 0.00 | collapsed: 0.91 / 0.81 / 0.77 |

Cloning pays the VC table's +0.339 ΔCER as filter rejections — about half the yield on `mk`
and `it` — **and separates the voices less well**. Named mode holds the yield and lands
different-name pairs on the stranger floor, including in languages the TTS ablation never
covered (`mk`, `gu`). So 26 languages were **topped up rather than replaced**: three
additional named voices per phrase, the auto-mode clip kept. 12,190 of 15,675 new clips
(77.8%) passed the filter, and the median collapsed language went **0.81 → 0.63**, with only
`ps` and `sq` still above 0.75.

Seven languages were left alone — `am`, `ml`, `my`, `sd`, `si`, `te`, `sr`. Those fail for a
different reason: FLEURS has no config for Sinhala, so there is no judge floor, the flat 0.25
gate applies, and `si` accepts **0.5%** of its clips whatever voice they are in. Re-voicing
does not help (named mode: 0%). They need a floor measured from a non-FLEURS corpus.

```bash
python tts/build_lexicon_queue.py --min-count 2      # 90% of the lexicon is not phrases
python tts/synth_lexicon.py --engine scicom --device cuda:6
python tts/filter_lexicon_synth.py --device cuda:6   # ASR round-trip, per-language gate
python tts/voice_diversity.py --langs all --device cuda:6          # are these real voices?
python tts/build_diversity_probe.py && python tts/build_diversity_topup.py
python scripts/split_lexicon_synth.py --accepted ... --audio-root ...
python scripts/build_lexicon_synth_release.py --synth audio/lexicon_synth_v4 \
    --primary-root audio/lexicon_synth_v3
python scripts/push_lexicon_synth.py                 # HF_TOKEN from the box's .env
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
