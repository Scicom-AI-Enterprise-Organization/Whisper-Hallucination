#!/bin/bash
# Method x learning-rate grid on the winning mix (`all`, every train split, 47% blank).
#
#   bash train/grid.sh 6 A     # half the grid on GPU 6
#   bash train/grid.sh 7 B     # the other half on GPU 7
#
# LoRA alpha tracks rank (2r) so the effective scaling is constant and rank is the only thing
# changing. 1e-3 is deliberately absent: the first sweep diverged there (corpus loss 9.76, and
# its checkpoint hallucinates on 100% of silence), so the grid brackets the stable 2e-4.
set -u
cd /root/whisper-halluc-train
GPU=${1:?usage: grid.sh <gpu> <A|B>}
HALF=${2:?usage: grid.sh <gpu> <A|B>}
MIX=${MIX:-all}
STEPS=${STEPS:-1000}

# rank:lr  (rank 0 means a full fine-tune)
A_JOBS="32:1e-4 32:2e-4 32:5e-4 0:5e-6 0:1e-5 0:2e-5"
B_JOBS="64:1e-4 64:2e-4 64:5e-4 128:1e-4 128:2e-4 128:5e-4"
[ "$HALF" = "A" ] && JOBS="$A_JOBS" || JOBS="$B_JOBS"

for job in $JOBS; do
  r=${job%%:*}; lr=${job##*:}
  if [ "$r" = "0" ]; then
    name="v3_full_${MIX}_lr${lr}"; method=full; extra="--batch-size 4 --grad-accum 4"
  else
    name="v3_lora_r${r}_${MIX}_lr${lr}"; method=lora
    extra="--lora-r $r --lora-alpha $((r*2)) --batch-size 8 --grad-accum 2"
  fi
  if [ -f "runs/$name/eval.json" ] && [ -d "runs/$name/merged" ]; then
    echo "[skip] $name"; continue
  fi
  echo "=== $name ==="
  CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL .venv_train/bin/python train/finetune_whisper.py \
      --mix train/mixes/$MIX.jsonl --method $method --lr "$lr" --steps "$STEPS" \
      --out "runs/$name" $extra 2>&1 | grep -E "trainable params|train_loss|saved|CUDA out|Error"
  bash train/eval_run.sh "runs/$name" "$GPU"
done
