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
`tts/clone_scicom.py`, which is speaker-NAME conditioned and therefore untargeted.

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

## VC selection, as measured

46 scorable lexicon phrases × 4 target speakers (2 Malaysian-Emilia, 2 LibriSpeech) = 184
clips per system. `tts/score_vc.py`, summary in `tts/vc_scores_summary.json`, figure from
`tts/plot_vc.py`.

CER alone picks the wrong winner: a converter that returns its input scores a perfect 0
degradation and is worthless. So the table reports **ΔCER against the source clip** plus
speaker similarity in both directions; the **gap** between them is the conversion that
actually happened. Percentages are on a **calibrated** scale (`tts/calibrate_vc_sim.py`):
at the clips' own duration (1.6 s), WavLM-sv scores 0.605 between different speakers and
0.853 between two clips of the same one, so 0% = a stranger, 100% = the target.

| | n | langs | CER | ΔCER | → target | ← source | gap | licence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Multilingual-Expressive-1.7B** *(untargeted)* | 138/138 | **21** | **0.395** | **+0.053** | — | **13%** | — | ours |
| kNN-VC | 184/184 | 20 | 0.441 | +0.099 | 68% | 59% | +9 | MIT |
| OpenVoice v2 (20 s ref) | 184/184 | 19 | 0.457 | +0.115 | 62% | 56% | +6 | MIT |
| OpenVoice v2 (6 s ref) | 184/184 | 20 | 0.467 | +0.125 | 61% | 53% | +8 | MIT |
| seed-vc (6 s ref) | 184/184 | 20 | 0.472 | +0.129 | 68% | 49% | +19 | GPL-3.0 |
| seed-vc (20 s ref) | 184/184 | **21** | 0.549 | +0.207 | 79% | 49% | +29 | GPL-3.0 |
| Higgs Audio v3 *(cloning)* | 184/184 | 20 | 0.654 | +0.312 | **84%** | **12%** | **+72** | non-commercial |
| OpenVoice + MeloTTS *(cloning)* | 40/184 | **4** | 0.077 | −0.434 | 69% | 19% | +50 | MIT |
| CosyVoice 2 | **2/184** | — | — | — | — | — | — | Apache-2.0 |

**Use Multilingual-Expressive-TTS-1.7B for the positive pool.** ΔCER +0.053 is half the best
converter's, 21 languages, and 13% source retention means it leaves the original voice
entirely. It conditions on a speaker NAME from `Scicom-intl/ExpressiveSpeech`, not a
reference waveform, so it cannot render a *given* target — irrelevant when the requirement
is "different, intelligible voices", decisive if you need a named speaker.

**Use Higgs Audio v3 if you need a named target.** The only system that genuinely transfers
identity (+72 gap vs ≤29 for everything else). Costs the most intelligibility (+0.312) and
its licence forbids using outputs to train non-Boson speech models — measurable, not usable
for the corpus. Best usable-licence targeted system is **seed-vc (6 s ref)**.

**kNN-VC and OpenVoice barely convert.** A +6 to +9 gap means the output sits almost equally
close to source and target: something in between, not the target's voice. Their good ΔCER is
partly explained by not changing much. Reference length is a real dial — seed-vc at 20 s
buys +29 identity for 0.078 more CER.

**Nobody reaches the ceiling** (best 84%), and the floor/ceiling distributions overlap
heavily at 1.6 s. Read these as "a different voice", not "this specific speaker".

**Cloned TTS is a coverage story.** MeloTTS covers 4 of 21 languages; CosyVoice's frontend is
zh/en/ja/ko. Conversion over a multilingual TTS covers the language range; cloned TTS does not.

**CosyVoice 2 could not be measured.** Its flow encoder dies with **SIGFPE** — a signal, so
Python cannot catch it — most often when the source is longer than the speaker prompt. Long
reference, short reference, single-threaded BLAS, chunked restarts and a pre-CosyVoice3
checkout all still crash within a few conversions; one process once completed 80.
`tts/setup/run_cosyvoice_resumable.sh` grinds through restarts if it is ever worth retrying.

**`phrases.jsonl` has 51 entries but only 49 distinct phrases** (`thanks for watching` and
`thank you for watching` appear twice). The VC Writer originally keyed dedup on
`(target, phrase)`, so systems calling `skip()` silently rendered fewer clips than those that
did not. It now keys on the source clip path, and `score_vc.py` deduplicates so every system
is scored on the same pairs.

**The degenerate lexicon entries are in this phrase set too** — `त र`, `त ह`,
`સ સ સ સ સ સ સ`. `score_vc.py` excludes them (3 phrases × 4 targets = 12 rows per system).
They are the same entries that gave `bn` CER 22.9 in the TTS table below, which was *not*
re-run with the filter — that column is still distorted.

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
