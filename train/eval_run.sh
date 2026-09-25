#!/bin/bash
# Score one trained checkpoint on BOTH halves of the benchmark.
#
#   bash train/eval_run.sh runs/v3_lora_all 7
#
# Arms, and what each answers:
# The COMPLETE benchmark -- all eight published arms plus lexicon_synth and wild:
#   silence music nonspeech   did hallucination fall?
#   reduplication             did looping fall -- and did it start deleting instead?
#   speech_in_noise           the hard middle: real speech as evidence weakens
#   genuine genuine_isolated  Malaysian speech, and the bare-phrase case
#   librispeech_test_clean    the English accuracy that must not move
#   fleurs                    the same guard, multilingual -- librispeech is English only
#   lexicon_synth (test)      can it still transcribe the phrases it used to invent?
#   wild (test)               all of that, on real audio rather than built stimuli
#
# --overwrite is not optional. run_benchmark skips an arm whose jsonl already exists, so a
# re-trained checkpoint written to the same run directory would be "scored" against the
# previous checkpoint's transcripts, silently.
set -u
cd /root/whisper-halluc-train
RUN=${1:?usage: eval_run.sh <run dir> [gpu]}
GPU=${2:-7}
MODEL="$RUN/merged"
[ -d "$MODEL" ] || { echo "no merged model at $MODEL"; exit 1; }

CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL .venv_wild/bin/python bench/run_benchmark.py \
    --model "$MODEL" --device cuda:0 --batch-size 24 --overwrite \
    --arms silence music nonspeech reduplication speech_in_noise genuine genuine_isolated \
           librispeech_test_clean fleurs lexicon_synth wild \
    --out "$RUN/bench" 2>&1 | grep -E "^\[done\]|^\[run \]|Error|Traceback" | tail -10

.venv_wild/bin/python bench/score_eval_run.py --run "$RUN" 2>&1 | tail -20
