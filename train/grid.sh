#!/bin/bash
# One or more grid jobs on ONE gpu. Jobs are `mix:rank:lr[:targets]`, or `rank:lr` to use $MIX.
# Rank 0 means a full fine-tune. `targets` is `attnmlp` (default) or `attn`.
#
#   bash train/grid.sh 0 all:32:1e-4 no_synth:8:1e-4:attn
#
# `attnmlp` adapts q/k/v/out_proj plus fc1/fc2; `attn` leaves the MLP alone, which is the
# original LoRA recipe and roughly halves the trainable count at every rank.
#
# Each job trains then immediately evaluates, so a finished GPU is never idle waiting for the
# rest of the grid. Already-complete jobs are skipped, which makes the whole thing resumable;
# FORCE=1 re-runs them anyway.
set -u
cd /root/whisper-halluc-train
GPU=${1:?usage: grid.sh <gpu> <[mix:]rank:lr> [...]}
shift
MIX=${MIX:-all}
STEPS=${STEPS:-1000}
FORCE=${FORCE:-0}
[ -f .env ] && set -a && . ./.env && set +a      # WANDB_API_KEY, HF_TOKEN

for job in "$@"; do
  # mix:rank:lr, or rank:lr with the mix from the environment
  tgt=attnmlp
  case "$job" in
    *:*:*:*) mix=${job%%:*}; rest=${job#*:}; r=${rest%%:*}; rest=${rest#*:}
             lr=${rest%%:*}; tgt=${rest##*:} ;;
    *:*:*)   mix=${job%%:*}; rest=${job#*:}; r=${rest%%:*}; lr=${rest##*:} ;;
    *)       mix=$MIX; r=${job%%:*}; lr=${job##*:} ;;
  esac
  if [ "$tgt" = "attn" ]; then
    mods="q_proj k_proj v_proj out_proj"; fam=loraattn
  else
    mods="q_proj k_proj v_proj out_proj fc1 fc2"; fam=lora
  fi
  if [ "$r" = "0" ]; then
    name="v3_full_${mix}_lr${lr}"; method=full; extra="--batch-size 4 --grad-accum 4"
  else
    name="v3_${fam}_r${r}_${mix}_lr${lr}"; method=lora
    extra="--lora-r $r --lora-alpha $((r*2)) --batch-size 8 --grad-accum 2"
    extra="$extra --lora-target-modules $mods"
  fi
  if [ "$FORCE" != "1" ] && [ -s "runs/$name/eval.json" ] && [ -d "runs/$name/merged" ]; then
    echo "[skip] $name"; continue
  fi
  echo "=== $name (gpu $GPU) ==="
  CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL .venv_train/bin/python train/finetune_whisper.py \
      --mix train/mixes/$mix.jsonl --method $method --lr "$lr" --steps "$STEPS" \
      --out "runs/$name" $extra 2>&1 \
      | grep -E "trainable params|train_loss|'loss'|saved|wandb|CUDA out|Error"
  bash train/eval_run.sh "runs/$name" "$GPU"
  echo "=== done $name ==="
done
