#!/bin/bash
# FLEURS for the five published checkpoints the README's tables are built from.
#
#   bash train/fleurs_published.sh 6
#
# librispeech is English. These five are compared on Malay, Chinese and Tamil throughout the
# README with no multilingual accuracy number anywhere, which FLEURS fixes: 20 clips per
# language, from the same cached sample the sweep uses.
set -u
cd /root/whisper-halluc-train
GPU=${1:?usage: fleurs_published.sh <gpu>}

MODELS="openai/whisper-large-v2 openai/whisper-large-v3 openai/whisper-large-v3-turbo \
mesolitica/malaysian-whisper-large-v2 mesolitica/Malaysian-whisper-large-v3-turbo-v3"

for m in $MODELS; do
    echo "=== fleurs $m (gpu $GPU) ==="
    CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL .venv_wild/bin/python bench/run_benchmark.py \
        --model "$m" --device cuda:0 --batch-size 16 --arms fleurs --out bench/results \
        2>&1 | grep -E "^\[done\]|^\[fleurs\] [0-9]|Error|Traceback" | tail -4
done

.venv_wild/bin/python bench/score_benchmark.py --results bench/results \
    --lexicon lexicon/combined_lexicon.csv --out bench/scores.json 2>&1 | grep -E "fleurs|->"
