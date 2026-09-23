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
already eaten `.env`, `.venv_bench`, a synth manifest mid-session, and (2026-09-18)
`lexicon/mined_lexicon.csv` + `lexicon/translated_lexicon.csv` — 15 minutes of NLLB
translation — because they were generated **on the box inside a tracked directory**. Anything
produced remotely that belongs in git must be pulled to the laptop *before* the next sync, not
left on the box to be reconciled later. Current excludes
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

**Padding a clip with digital zeros is almost a no-op — say so before measuring it.** Whisper
pads every input to a 30 s log-mel window, so a 0.9 s clip already sits in ~29 s of zeros:
trailing zeros change nothing by construction, and leading zeros only shift where the speech
starts. `run_benchmark.py --pad-kind roomtone` pads with real near-silence from the `silence`
arm instead, which is a different signal — a noise floor these checkpoints demonstrably
hallucinate over. Keep the zeros arm as the control; it moved mean CER by 0.01–0.02 where room
tone moved it by 0.38 (`malaysian-whisper-v2`) and 2.24 (`Malaysian-turbo-v3`).

**`lexicon_synth` inverts the benchmark's ranking, which is the point.** On the negative arms
`Malaysian-turbo-v3` is far the best (90.6% genuinely empty on voice-free audio). On 6,267
clips where the phrase IS spoken it recovers 35–41% against large-v3's 68–70%, and
`malaysian-whisper-v2` manages 30%. The quiet is a prior against emitting text, not a
detector. Always report both sides; either alone picks the wrong checkpoint.
Harness: `bench/run_benchmark.py --arms lexicon_synth`, scorer
`bench/score_lexicon_synth.py` (per-language gates from `tts/judge_floor.json`), figure
`bench/plot_lexicon_synth.py`. 15 runs (5 models × 3 conditions) take ~90 min over GPUs 6-7.

**Mean CER on this arm is a tail statistic.** Median CER moves by hundredths between
conditions while the mean triples, because 3–6% of clips run past 2× the phrase length.
Report `cer_median` and `over_2x_rate` together, or the number says nothing about the typical
clip. The tail costs wall-clock too: same 6,267 clips, `malaysian-whisper-v2` 496 s → 627 s
under room tone while large-v3 stays flat.

**`loop_rate` (run ≥ 6) is confounded on the reduplication arm.** A correct transcript of
six repeats is itself a run of six, so the metric fires on right answers: large-v3 goes
1.7% → 37.9% between `n_repeats` 5 and 6 with no change in behaviour. On that arm read
`runaway_rate` (emitted/true > 1.5) and `empty_rate`; keep `loop_rate` for the speech arms.
`bench/reduplication_profile.py` slices the arm by its four stimulus knobs (on the box, it
needs `bench/results/`), `bench/plot_reduplication.py` draws it. Findings: laughter is the
trigger (fine-tunes 51–56%, clicks 0%), more repeats → more runaway for every model, and
trailing silence does not raise it.

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

**`.venv_omni`** — OmniVoice (646 languages, ~9× faster than the others). **Its code is
Apache-2.0 but the released weights are CC-BY-NC**, "due to constraints from its training data
(e.g. Emilia)" — the model card says so plainly and this repo said Apache-2.0 for months.
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

**The manifest's `voice` column is a label, not a measurement.** OmniVoice takes no speaker
argument at all, so every one of its clips carries `voice=""` — and `tts/voice_diversity.py`
(WavLM-sv x-vectors, same calibrated scale as the VC table: 0.605 different speakers, 0.853
same speaker, both at 1.6 s) found **33 of its 74 languages at a median pairwise cosine of
0.75 or worse** — mk 0.91, fo/tg 0.89, az/sq 0.88 — i.e. one voice per language. The
Multilingual-Expressive half is genuinely multi-speaker and the measurement proves the
conditioning lands: same-name pairs 0.826, different-name 0.601.

**The fix is speaker names, not cloning.** `tts/build_diversity_probe.py` ran both routes over
the same 4 collapsed languages × 24 phrases × 4 voices, scored in one venv:

| route | yield mk / gu / it | median CER | different-voice cosine |
|---|---|---:|---|
| Multilingual-Expressive, speaker name | **86% / 80% / 79%** | 0.07 / 0.17 / 0.00 | **0.652 / 0.638 / 0.578** |
| OmniVoice, reference cloning | 46% / 68% / 53% | 0.32 / 0.25 / 0.26 | 0.684 / 0.722 / 0.687 |
| *(auto mode, for reference)* | 89% / 73% / 83% | 0.02 / 0.11 / 0.00 | collapsed: 0.91 / 0.81 / 0.77 |

Cloning pays the VC table's +0.339 CER as filter rejections — about half the yield on mk and
it — **and separates the voices less well**. Named mode holds yield and lands different-name
pairs on the stranger floor. It also works in languages the TTS ablation never covered (mk,
gu), so "untagged coverage" is not the same as "no coverage". `synth_lexicon.py` grew
`ref_audio`/`ref_text` per queue item for the cloning arm; it stays, but the top-up
(`tts/build_diversity_topup.py`) runs named mode, keeps the auto-mode clips, and adds 3 voices
per phrase.

**The top-up, as run.** `tts/build_diversity_topup.py` queued 3 named voices for each of
5,225 phrases over 26 languages (15,675 clips, 0 failures, ~4.3 clip/s per GPU); the filter
kept **12,190 / 15,675 = 77.8%** (ka 94%, tr 93%, mk 90% at the top; fo/nn 33%, mt 48% at the
bottom). Pooled over both engines the median collapsed language moved **0.808 → 0.629** —
mk 0.905 → 0.576, af 0.806 → 0.521 — with only `ps` and `sq` still at 0.75+. The corpus went
16,922 → **29,112 clips** (train 22,845 / test 6,267, 15.8 h), non-English share 86% → 92%.

Merging two synthesis roots needed three fixes, all of which bite again next time:
`split_lexicon_synth.py` carries a `corpus_root` per clip and takes `--audio-root`, because
the v3 filter wrote its manifests into `lexicon_synth_v3_{sconly,omonly}` while the audio
stayed in `lexicon_synth_v3` — deriving the root from the manifest's own directory points at
nothing. `build_lexicon_synth_release.py` resolves audio and `meta` per root and takes
`--primary-root`, whose clips keep bare ids so the published rows are not all renamed; every
other root is tagged, because `idx` restarts at 0 in each one and would collide.
`voice_diversity.py --pool-engines` measures a language across engines and roots, which is
what the shipped corpus actually is.

**`sync --delete` ate `bench/wild_results/` WHILE the job was writing to it.** The baseline
transcribed all 3,607 HALAS clips at 10.9 clip/s and then died on
`FileNotFoundError: bench/wild_results/whisper-large-v2.summary.json` -- the directory was
created at startup and removed by a `sync` fired minutes later for an unrelated file. A
running job's output directory is exactly as vulnerable as a finished one. Excludes now cover
`audio_wild`, `bench/wild_results`, `.venv_wild`.

**`sync --delete` ate `tts/voice_diversity_after.json` mid-session** — a 7-minute GPU scan,
written on the box inside a tracked directory, deleted by the next `sync` before it was
pulled. That is the third time. The excludes now carry `tts/voice_diversity*.json` and
friends, but the rule is the rule: **add the exclude before the job writes, not after.**

**`.venv_omni` has no `pyarrow`.** The release builder needs pyarrow + datasets + soundfile,
and only `.venv_bench` has all three; `.venv_omni` is the scoring venv (transformers, torch,
WavLM). Building the parquet from the wrong venv fails at import, after the split has already
been rebuilt.

**`si` and `mg` fail for a different reason.** FLEURS has no config for either, so there is no
judge floor, the flat 0.25 gate applies, and Sinhala accepts **0.5%** of auto-mode clips at a
median CER of 0.915. Re-voicing does not help (named mode: 0%). Those languages need a floor
measured from a non-FLEURS corpus, or an explicit unvalidatable flag; the top-up skips them
along with am/ml/my/sd/te (yield already ~0).

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
| **OmniVoice** *(cloning)* | 183 | **21** | 0.683 | +0.339 | **107%** | 15% | **+92** | 1.28 | 1% | code Apache-2.0, weights CC-BY-NC |
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
  Among the openly-released options it is a trade: **OmniVoice** (code Apache-2.0, weights
  CC-BY-NC) gives the strongest
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
| OmniVoice | 0.349 | 0.018 | 3 | code Apache-2.0, **weights CC-BY-NC** | **97/100 langs** |
| Higgs v3 | 1.230 | 0.484 | 1 | non-commercial | 82/100 |
| Higgs v2 | 1.089 | 0.713 | 0 | non-commercial | — |

**Two more candidates, measured 2026-09-22 in one scoring run with the other four.**

| | mean CER | wins | languages | different-voice cosine |
|---|---:|---:|---:|---:|
| OmniVoice (voice design, `instruct`) | 0.589 | 3 | 646 | **0.453** |
| ToucanTTS | 0.830 | 0 | **7,233** | 0.738 |

`instruct` is a **controlled vocabulary of 48 tags**, not prose — gender, age, pitch, accent,
whisper, plus Chinese dialects, in mutually exclusive groups (`_INSTRUCT_MUTUALLY_EXCLUSIVE`
in `omnivoice/models/omnivoice.py`). Free text raises `ValueError`. Use
`"female, young adult, high pitch"`, not a sentence. It is the **best diversity-per-CER trade
measured**: 0.453 different-voice cosine (below the 0.605 stranger floor) for +0.240 CER over
auto mode, against reference cloning's +0.339 for 0.68–0.72. Worth revisiting for the next
top-up instead of the named-speaker route.

**ToucanTTS is breadth without fidelity.** 7,233 languages via articulatory features, and it
wins nothing: CER 0.830, and its WGAN-sampled speaker embeddings sit at 0.738, barely apart.
Install notes: `.venv_toucan` (its own uv venv), requirements pins (torch 2.4 / numpy 1.23 /
librosa 0.9) all have to be dropped, `sounddevice` is imported at module level for playback
and needs PortAudio the box lacks — stub the module rather than fixing apt — and `phonepiece`
imports `pip`, so install `pip` into the venv. `ControllableInterface.read()` returns
`(sr, wav, figure_path)`, three values, not two.

Use **OmniVoice** for the 100-language sweep (coverage is enumerable and it is the fastest
by far; note the weights are CC-BY-NC even though the code is Apache-2.0) and **Multilingual-Expressive** for ms/en/zh/ta. Higgs is out on both counts — its
licence independently forbids using outputs to train non-Boson speech models.

**The Rahman mistake reaches this table too.** One of the three speaker names in the grid
(`multilingual-tts_audio_Rahman`, 17 of 51 phrases) is not in `ExpressiveSpeech`, so those
clips are unconditioned: Grace 0.130, DisfluencySpeech 0.254, Rahman 0.280.
`tts/aggregate_tts.py --valid-speakers-only` drops those phrases *for every system* — the
systems have to be compared on the same text — leaving 32 clips over 20 languages:
Multilingual-Expressive **0.192** / 17 wins, OmniVoice 0.450 / 2, Higgs v2 1.158, Higgs v3
1.583. The ranking is unchanged and the gap widens, so the 22-language table stays the
headline; the plain run still reproduces it exactly (0.221 / 0.349 / 1.089 / 1.230, 18 wins).

## Wild audio — where the real failures are

`audio_wild/` holds real recordings that made a model fail, built by
`scripts/build_halas_arm.py` (human labels) and `scripts/mine_wild_hallucinations.py` (no
labels needed). `bench/run_wild_baseline.py` runs checkpoints over it, `bench/score_wild.py`
joins the outputs back to the manifests. Published as the `wild` config.

**Mine spontaneous audio, not curated corpora.** Yield per 8,000 clips heard: AMI meetings
235, Earnings-22 31, VoxPopuli 4, People's Speech 1. Read speech recorded for a dataset almost
never triggers this; multi-party audio with real silence between turns does.

**Two labels are free, the third is not.** A token run ≥ 6 and "words over VAD-confirmed
silence" need no annotator. `lexicon_hit` on its own is worthless — every meeting is full of
genuine "Yeah." — so it only counts alongside a blank verdict.

**An RMS threshold is not a silence test.** At −45 dBFS it flagged 2,920 of 6,000 AMI clips:
AMI headset mics record at −45 to −51 dBFS while containing perfectly good speech. Silero VAD
replaced it, and hit rate fell 49% → 2.9%. **And a starved VAD is not silence either**: below
~250 ms Silero returns `speech_frac 0.0` regardless, so `blank_speech` requires ≥ 0.4 s.

**HALAS joins on `{segment_id}_{file_id}.wav`**, the pair distil-whisper/earnings22's `chunked`
parquets carry as two columns. 3,607 of 3,611 matched, and the per-model flag counts reproduce
the published table (v2 1578 vs 1581, phi4 1096 vs 1096, large-v3 857 vs 858) — that agreement
is the check that the join is right. Count flags with exact membership, not `in` on the joined
string: "canary" is a prefix of "canary_flash" and "whisper_large_v3" of "..._turbo", which
inflated canary to 1,651.

**What the wild arm measured.** On 263 VAD-confirmed voice-free clips every OpenAI checkpoint
emits words on 100%, `Malaysian-turbo-v3` on 8.0% — the synthetic non-speech result, harder.
Against HALAS human labels, CER separates flagged from clean by +0.22 (large-v3) while the
lexicon rate separates them by 3.8 points, which is the measured case against a text blocklist.
Aphasia is the worst trigger: `Malaysian-turbo-v3` loops on 25.1% and emits nothing on 59.9%.

**The aphasia clips are never re-hosted.** Clinical speech from a membership-gated corpus;
`build_wild_release.py` only reads `audio_wild/`, and they live under `audio/`, so exclusion is
structural rather than a flag someone can forget.

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
