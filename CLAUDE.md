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
