#!/bin/bash
# Run the FLEURS arm over checkpoints that were benchmarked before the arm existed.
#
#   bash train/fleurs_pass.sh <gpu> <run dir> [run dir ...]
#   bash train/fleurs_pass.sh 0 BASE            # the untuned openai/whisper-large-v3
#
# Every run reads the SAME cached sample (bench/fleurs_sample.parquet), so the column is
# comparable across the grid even though the runs were trained and scored days apart.
# Each checkpoint is re-scored afterwards, which folds `fleurs` into its existing eval.json
# without touching the other arms.
set -u
cd /root/whisper-halluc-train
GPU=${1:?usage: fleurs_pass.sh <gpu> <run dir> [...]}
shift

for RUN in "$@"; do
    if [ "$RUN" = "BASE" ]; then
        MODEL=openai/whisper-large-v3
        OUT=bench/results          # where the five published checkpoints already live
    else
        MODEL="$RUN/merged"
        OUT="$RUN/bench"
        [ -d "$MODEL" ] || { echo "skip $RUN: no merged model"; continue; }
    fi
    echo "=== fleurs $RUN (gpu $GPU) ==="
    CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL .venv_wild/bin/python bench/run_benchmark.py \
        --model "$MODEL" --device cuda:0 --batch-size 16 --arms fleurs --out "$OUT" --overwrite \
        2>&1 | grep -E "^\[done\]|^\[fleurs\]|Error|Traceback" | tail -5
    [ "$RUN" = "BASE" ] || .venv_wild/bin/python bench/score_eval_run.py --run "$RUN" 2>&1 | tail -3
    echo "=== done fleurs $RUN ==="
done
