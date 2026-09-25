#!/bin/bash
# The training sweep. RUNS ON THE BOX, GPUs 6-7.
#
#   bash train/sweep.sh stage1     # LoRA over every mix, whisper-large-v3
#   bash train/sweep.sh stage2 MIX # best mix on turbo, then full fine-tunes
#
# The objective is a trade, not a single number: cut hallucination and looping WITHOUT losing
# accuracy. Every run is therefore scored on both halves of the benchmark, and a run that wins
# the non-speech arms by deleting speech is a loss.
set -u
cd /root/whisper-halluc-train
PY=.venv_train/bin/python
STAGE=${1:-stage1}
MIXES="blank_only corpus plus_synth all balanced synth_heavy"

train_one () {                      # model, mix, method, out, gpu
  local model=$1 mix=$2 method=$3 out=$4 gpu=$5
  if [ -d "$out/merged" ]; then echo "[skip] $out"; return; fi
  echo "=== train $out ==="
  CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL $PY train/finetune_whisper.py \
      --mix train/mixes/$mix.jsonl --model "$model" --method "$method" --out "$out" \
      --steps "${STEPS:-1000}" --batch-size "${BS:-8}" --grad-accum "${GA:-2}" 2>&1 | tail -6
}

case "$STAGE" in
  stage1)
    for mix in $MIXES; do
      train_one openai/whisper-large-v3 "$mix" lora "runs/v3_lora_$mix" "${GPU:-7}"
    done
    ;;
  stage2)
    BEST=${2:?usage: sweep.sh stage2 <mix>}
    train_one openai/whisper-large-v3-turbo "$BEST" lora "runs/turbo_lora_$BEST" "${GPU:-7}"
    train_one openai/whisper-large-v3       "$BEST" full "runs/v3_full_$BEST"    "${GPU:-7}"
    train_one openai/whisper-large-v3-turbo "$BEST" full "runs/turbo_full_$BEST" "${GPU:-7}"
    ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
