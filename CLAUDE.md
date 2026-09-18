# CLAUDE.md

Working notes for this repo. Everything here was learned the hard way — read before
re-deriving.

## Two artifacts, kept strictly separate

| | what | where |
|---|---|---|
| **Benchmark** | evaluation only, `split="test"` | published: `Scicom-intl/Whisper-Hallucination` |
| **Training corpus** | separate build, disjoint by construction | box: `/root/whisper-halluc-train/corpus/` |

**Never train on the benchmark.** Every audio config is `test` for that reason. Disjointness
is structural, not a promise:

| pool | benchmark | training |
|---|---|---|
| FSD50K | `eval` split | `dev` split |
| FMA | shards 0–1 | shards 2–12 |
| Malay speech | sarawak / manglish / fleurs | Malaysian-Emilia |
| silence, reduplication | seed 0 | seed 1000, held-out CV units |

`benchmark/exclusions.json` is the belt-and-braces check on top.
`scripts/verify_disjoint.py` must report **0 collisions** before any training run.

## Hard rules

- **Whisper/TTS inference runs on the box, never the laptop** — via `claude-ping`, even for
  a five-clip debug script.
- **GPUs 6–7 only.** 0–5 are other people's jobs at 100% utilisation.
- Published dataset stays permissively licensed (CC BY 4.0 / CC0 / MIT / Apache-2.0).
  NC sources (ESC-50, UrbanSound8K, malaysia-ai/*, Emilia) are *fetched by script*, never
  re-hosted — that keeps the ablation reproducible without relicensing anything.

## claude-ping

Config is `claude-ping.json` (box 1023, key at `../scicom/dataset/scicom` — outside this
repo, don't move it). `claude-ping up` once, then `exec` / `sync` / `run`.

**`sync` is `rsync --delete` and WILL destroy anything not in `sync_excludes`.** It has
already eaten `.env`, `.venv_bench`, and a synth manifest mid-session. Current excludes
cover `.venv*`, `.env`, `audio`, `audio_train`, `corpus`, `tts/out`, `bench/scores.json`.
**Add to that list before generating anything new on the box.** Restore secrets with
`claude-ping env-sync`, never by hand.

`scp` fails while the master connection is up ("Connection closed"). Write remote files with
`claude-ping exec 'cat > path <<EOF'` instead.

Long jobs go through `claude-ping run --session <name>` (detached tmux, survives
disconnection), not `nohup` over a one-shot `exec`.

## Traps that cost time

**Nested heredocs through `claude-ping exec`.** Quotes get mangled and have silently created
files named `PY` and `{r[hyp][:80]!r})`. Write the script to a local file, `claude-ping
sync`, then run it. Do not inline anything with nested quotes.

**`pgrep -f <name>` matches its own shell.** A waiter loop like
`while pgrep -f build_x; do sleep 30; done` never exits, because the loop's own command line
contains `build_x`. Several of these accumulated and exhausted the laptop's RAM. Gate on a
done-file or `claude-ping watch` instead.

**`datasets >= 5` needs `torchcodec` to both encode AND decode audio.** Neither is worth
installing:
- *Writing*: build the parquet with pyarrow directly — `audio` as
  `struct<bytes: binary, path: string>` plus schema metadata
  `b"huggingface": {"info": {"features": Features(...).to_dict()}}`. `Features.to_dict()`
  works without torchcodec. See `scripts/build_hf_release.py`.
- *Reading*: `ds.cast_column("audio", Audio(decode=False))` then `soundfile` on the bytes.

**Parquet dictionary encoding inflates FLAC columns ~1.65×.** Every value is unique and
already compressed, so the dictionary overflows and falls back to PLAIN, storing the payload
twice. Pass `use_dictionary=[]` for any table with an audio column.

**Per-shard counters silently overwrite files.** A counter reset inside the shard loop gave
4,694 manifest rows pointing at 2,908 files — ~1,786 rows had *mismatched audio and text*.
Caught only because the parquet was larger than the FLAC directory. Counters belong per
source, not per shard; assert `len(rows) == len(set(paths)) == files_on_disk`.

**Metric ambiguity is real, not pedantic.** Published hallucination rates for the same model
on the same corpus differ >10× (99.97% vs 76.08%) purely over whether Whisper's own
no-speech filter runs first. The scorer reports `hallucination_rate` (word content) and
`hallucination_rate_any_output` (a bare `"."` counts) — on silence those are 61.9% and
100.0% for large-v3. Always say which you mean. vLLM's `verbose_json` doesn't return
`no_speech_prob` at all, so the filtered variant is a research number only.

**The lexicon contains degenerate entries.** `સ સ સ સ સ સ સ`, `त र` — Whisper loop artefacts
captured as phrases. They wreck CER comparisons (one gave CER 22.9). Filter them:
single repeated token, or ≤2 distinct characters.

## Production is vLLM

faster-whisper can't serve at scale, so most of `ablation/configs.json` is diagnostic only.
vLLM does **not** implement Whisper's transcription loop — no `no_speech_threshold`,
`compression_ratio_threshold`, `logprob_threshold`, `condition_on_previous_text`,
temperature fallback, or `no_repeat_ngram_size`. It exposes `temperature`, `top_p`, `top_k`,
`min_p`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, and
(inefficient) beam search.

**The cascade cannot happen on vLLM.** It chunks into independent 30 s windows and doesn't
carry text across them (open request, vLLM PR #20249), so
`condition_on_previous_text=False` — the single largest lever elsewhere — is already in
effect for free. What remains is *within-window* runaway, bounded directly by `max_tokens`.
Expect the vLLM baseline to loop less than a faster-whisper baseline; that's architecture,
not model quality.

Because the gating knobs are gone, leverage lives outside the engine: VAD front-end,
chunking policy, and a post-hoc filter (`phrases/ban_candidates.csv`).

## Model-specific setup

Each of these needs its **own venv** (`uv venv --system-site-packages`); their torch and
torchao pins conflict.

**`.venv_bench`** — Whisper benchmark + neucodec + Scicom TTS.
neucodec → torchtune → `from torchao.dtypes.nf4tensor import NF4Tensor`, removed in torchao
0.18; pinning back to 0.12 breaks against torch 2.14. Neither version satisfies both, so
`tts/nf4_shim.py` registers a placeholder (NF4 is LLM quantisation, the codec path never
touches it). Import it **before** neucodec. NeuCodec itself is gated upstream — load the
`Scicom-intl/neucodec` mirror via `tts/load_neucodec.py`, because the upstream loader
asserts the repo id.

**`.venv_higgs3`** — Higgs Audio v3.
`bosonai/higgs-tts-3-4b` is `model_type: higgs_multimodal_qwen3`, unsupported by vanilla
transformers (Boson points at SGLang-Omni). Use
`multimodalart/higgs-audio-v3-tts-4b-transformers` — byte-identical weights (927 tensors)
plus a modeling file — with `trust_remote_code=True`. Needs `torchaudio`.
- `audio_head.weight MISSING` in the load report is **benign**: it's in `_tied_weights_keys`
  and gets tied to `audio_embedding.weight`. Not randomly initialised.
- There is no `.generate()`. Use `generate_speech(text, tokenizer, ...)` — **text first** —
  which returns a mono 24 kHz waveform.

**`.venv_omni`** — OmniVoice (Apache-2.0, 646 languages, ~9× faster than the others).
- Language ids are ISO-639-3 (`arb`, `npi`, `fil`), not the lexicon's `ar`/`ne`/`tl`. An
  unmapped code does **not** error — it silently drops to language-agnostic mode. Resolve
  with `tts/omnivoice_langmap.py`; `la`, `fo`, `su` genuinely aren't supported.
- **One bad phrase raises for the whole batch** (`zero-size array to reduction`), costing 8
  clips per failure. `synth_omnivoice.py` falls back to per-item on batch error: 43/51 → 51/51.

**Higgs Audio v2** (in `.venv_bench`) needed three corrections: the chat template requires a
system message, the processor accepts only `return_tensors="pt"`, and generate returns
codebook tokens where **1024/1025 are BOS/EOS markers** that must be stripped before
`audio_tokenizer.decode` (which also has to be moved to the GPU).

## Voice conversion / cloning candidates

Four more venvs, same reason as the TTS ones — `tts/setup/install_vc.sh <name>` builds each:

**`.venv_knnvc`** — kNN-VC. Three deps (torch, torchaudio, numpy); the repo vendors WavLM and
torch.hub fetches WavLM-Large + the prematched HiFi-GAN on first run. The only candidate that
needs *minutes* of target audio rather than one clip — its "speaker" is a pile of WavLM frames
to match against, which is also why it should be the most language-agnostic.

**`.venv_seedvc`** — seed-vc. `requirements.txt` opens with four `--pre --index-url nightly`
lines that fight the pinned versions below them; drop those, keep the pins. Drive it through
`seed_vc_wrapper.SeedVCWrapper`, not `inference.py` — the CLI reloads every model per call.
It resolves configs relative to cwd, so `os.chdir(repo)` before constructing it.

**`.venv_openvoice`** — OpenVoice v2. Their `numpy==1.22` / `librosa==0.9.1` pins have no
wheels for modern Python; the modern pair works. Skip `se_extractor` (it runs
whisper-timestamped just to segment a reference) and call `ToneColorConverter.extract_se`
directly. Kill the watermark (it perturbs the audio the scorer measures) by setting
`conv.watermark_model = None` after construction — the documented `enable_watermark=False`
kwarg is forwarded to a parent that does not accept it and raises.
Checkpoints are `myshell-ai/OpenVoiceV2`, `converter/*` only.

**`.venv_higgs3`** also drives the cloning arm (`tts/clone_higgs3.py`): v3 takes
`reference_audio` + `reference_sample_rate` + `reference_text`, so unlike Scicom's
Multilingual-Expressive it can aim at a given speaker. `.venv_bench` drives
`tts/clone_scicom.py`, which has both paths: `--mode clone` (reference audio, targeted)
and `--mode named` (speaker name; no reference clip, so nothing to score `→ target` against).

**`.venv_cosyvoice`** — CosyVoice 2. `openai-whisper`'s setup.py imports `pkg_resources`,
which **setuptools removed in 81** — so it needs `--no-build-isolation` *and* `setuptools<81`
in the venv. Not optional: `cosyvoice/cli/frontend.py` does `import whisper` at module level.
Needs
`third_party/Matcha-TTS` on `sys.path`, and it hardcodes `cuda:0`, so pick the GPU with
`CUDA_VISIBLE_DEVICES` **before importing torch**, not with a device argument.

New box paths — all added to `sync_excludes`, do that *before* generating anything:
`vc_repos/` (cloned repos + checkpoints), `tts/vc_targets/`, `tts/vc_out/`, `tts/vc_scores.json`.

**`${1:?usage: ... {a|b}}` in bash.** The expansion ends at the *first* `}`, so a usage message
containing braces leaks the rest into the variable — `install_vc.sh knnvc` set `sys=knnvc}`
and every system fell through to "unknown". No braces in `:?` messages.

## Synthesising the lexicon at scale

Pipeline: `tts/build_lexicon_queue.py` → `tts/synth_lexicon.py` → `tts/filter_lexicon_synth.py`,
with `tts/qa_lexicon_synth.py` for a stratified spot-check and `tts/build_voice_plan.py` for
per-language routing.

**90% of the lexicon is not phrases.** Of 39,476 queued entries, **35,594 were observed
exactly once**, and 27,974 of those run ≥5 words. They are raw ASR fragments, not things
anyone says:

    '0073a this vehicle s got 72 00 miles on it 3 5l v engine'
    '1 188 and 1 792 for the'

Digits split into separate tokens, no punctuation, truncated mid-sentence. Synthesising
`0 1 0 4` yields "zero one zero four" and transcribes back as `0104`, so the round-trip fails
by construction. On these, **both engines score CER ≈ 1.0 on English** (OmniVoice 0.651 mean,
Multilingual-Expressive 0.712, neither over-generating) — when two independent engines fail
identically the input is at fault. Synthesise `--min-count 2`: 3,882 entries across 85
languages, median 2 words. The rest is debris that will fail QA anyway.

**The queue is sorted by observation count, descending**, so the run degrades gracefully:
stopping early leaves the most-hallucinated phrases done rather than a random sample. That is
what made it safe to stop the English run at 7,424 of 30,608 — all 1,380 of its `count>=2`
entries were already complete.

**A voice plan built on short phrases does not generalise.** `lexicon_voice_plan.json` came
from 2-3 high-frequency phrases per language (`thank you`), where Multilingual-Expressive
scores 0.000 on several languages. On real lexicon phrases the same routing gives ta 1.49,
el 1.37, ur 1.10 with duration ratios past 2× — the same runaway its cloning path shows.
Re-measure routing on the phrases you will actually synthesise.

**Batching is worth 4.7× and is safe.** Batch 32 with left padding plus length-sorted windows
gives 8.53 clip/s across two GPUs against 1.17 at batch 8 one-at-a-time. A batch costs the
LONGEST generation in it, so `--sort-window` sorts by phrase length inside 512-item priority
windows; without it a 2-word phrase shares a budget with a 40-word one. `tts/check_batching.py`
verifies batched against single generation directly — **median CER 0.000 both ways** (the mean
differs by 0.036, which is sampling noise under `do_sample=True`). Worth re-running if the
generation path changes.

**`max_new_tokens` scales with the phrase**, ~22 tokens/word plus slack (NeuCodec is 50
tokens/s, speech ~2.5 words/s). A flat 1024 spends 20 s of budget on a two-word phrase, which
is both wasteful and exactly how the cloning path ran away.

**Force the language in every scorer.** Whisper auto-detecting on a short non-Latin clip lands
in the wrong language and returns CER > 1.0 that says nothing about the audio — it read `ar`
at 1.47 until the QA was fixed to force the language and batch by it, as `score_tts.py` and
`score_vc.py` already did.

**Piping a long run through `| tail -N` hides it until it exits.** Both `head` and `tail`
buffer, so progress lines never appear; `stdbuf -oL` on the producer and the filter, or no
pager at all.

## VC selection, as measured

46 scorable lexicon phrases × 4 speaker slots (2 Malaysian-Emilia + 2 LibriSpeech references
for the targeted systems; 4 named speakers for the name path) = 184 clips per system.
**OmniVoice is the source audio for every converter, so it is the baseline** — ΔCER is what a
system adds on top of the OmniVoice clip, and `← source` is similarity to the OmniVoice voice.
`OpenVoice + MeloTTS` is the exception: MeloTTS renders it, hence the negative ΔCER.
Being the baseline does **not** make it ineligible as a candidate: OmniVoice takes
`ref_audio`/`ref_text` too, so it clones to a target like Higgs v3 — `tts/clone_omnivoice.py`,
scored 2026-09-17. Conditioning on a voice costs it **+0.339 CER** over its own auto-mode
clip, and buys the strongest identity transfer in the table (107%, gap +92) at Apache-2.0.

**Scoring environment matters and was not pinned.** `.venv_bench` (the original scorer venv)
is gone from the box; re-scoring under `.venv_omni` (transformers 5.17.0, torch 2.11+cu128)
reproduces WavLM similarity, duration and langs *bit-identically* but moves Whisper CER on
one system — `higgs3_clone` 0.627 → 0.653, everything else within ±0.002. So **score every
arm in one environment**: a row scored in a different venv is not comparable. All rows above
were re-scored together; `tts/make_vc_summary.py` rebuilds the summary from `vc_scores.json`
afterwards (it used to be hand-made).

Scoring is otherwise **deterministic** — re-running the same audio in the same venv reproduced
11 of 12 systems bit-identically. The exception was `knnvc`, which moved 0.443 → 0.437 CER
between two identical runs, so treat differences under ~0.01 CER as noise rather than signal.
`tts/score_vc.py`, summary in `tts/vc_scores_summary.json`, figure from `tts/plot_vc.py`.

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
  Among permissive licences it is a trade: **OmniVoice** (Apache-2.0) gives the strongest
  identity in the table (107%, gap +92, 1.28×) for +0.339 ΔCER; **seed-vc (6 s ref)** keeps
  intelligibility (+0.129) and barely moves the voice (+19).

**OmniVoice reaches the ceiling too** (107%) — with the same caveat, since it runs 1.28×
long and the calibration assumed 1.6 s. The floor/ceiling distributions also overlap heavily
at 1.6 s. **Cloned TTS is a coverage story**: MeloTTS covers 4 of 21 languages and CosyVoice's
frontend is zh/en/ja/ko. **CosyVoice 2 could not be measured at all** — its flow encoder dies
with SIGFPE, which no `except` can catch.

**`phrases.jsonl` has 51 entries but only 49 distinct phrases** (`thanks for watching` and
`thank you for watching` twice). The Writer keys dedup on the source clip path, not the text —
keying on the phrase silently skipped repeats for systems that call `skip()`. `score_vc.py`
deduplicates so every system is scored on the same pairs, and the degenerate entries
(`त र`, `त ह`, `સ સ સ સ સ સ સ`) are excluded, 3 phrases × 4 slots = 12 rows per system.

## TTS selection, as measured

48 lexicon phrases, 22 languages, identical text, ASR round-trip CER
(`tts/score_tts.py`, results in `tts/tts_scores.json`):

| | mean | median | wins | licence | coverage |
|---|---:|---:|---:|---|---|
| Multilingual-Expressive-TTS-1.7B | **0.221** | **0.000** | **18** | ours | untagged |
| OmniVoice | 0.349 | 0.018 | 3 | **Apache-2.0** | **97/100 langs** |
| Higgs v3 | 1.230 | 0.484 | 1 | non-commercial | 82/100 |
| Higgs v2 | 1.089 | 0.713 | 0 | non-commercial | — |

Use **OmniVoice** for the 100-language sweep (coverage is enumerable and the licence is
clean) and **Multilingual-Expressive** for ms/en/zh/ta. Higgs is out on both counts — its
licence independently forbids using outputs to train non-Boson speech models.

## Upstream data caveats

- **Malaysian-Emilia's audio archive is incomplete.** It's a split ZIP64 where ~40% of
  members reference data absent from the two published parts. Python's `zipfile` also
  refuses spanned archives outright. `scripts/repair_spanned_zip.py` resolves each member's
  true offset *empirically* against the file rather than trusting the disk fields, recovering
  134,218 of 223,029. That's why phrase yield was 905, not the 4,000 requested.
- **Wikimedia throttles bots hard** (HTTP 429 at 6 workers). Lingua Libre harvesting is
  deliberately serial at ~1 req/s. The Hub takes 16 workers happily.
- **FLEURS is read Wikipedia prose** — 3,740 Malay utterances yielded exactly *one*
  `terima kasih`. Register matters more than size for conversational phrases.

## Known weaknesses of the current benchmark

Stated plainly so nobody over-claims:

- Positive pool is 52% synthetic TTS (Synth-Manglish) and 23% read prose; only 25% is real
  conversational, and that's Sarawak dialect.
- No telephony condition — everything is 16 kHz clean, production is 8 kHz narrowband.
- zh/ta positives are ~nil (11 and 4 clips).
- Malay hallucination counts are borrowed from public lexicons (6 phrases), not measured, so
  the Malay half of `ban_candidates` is unfounded until a local probe runs.
- `music` may contain singing (`vocals=unknown`); `nonspeech` is the label-verified
  alternative.
- `speech_in_noise` is synthetic mixing — exact SNR, but no Lombard effect or channel.
