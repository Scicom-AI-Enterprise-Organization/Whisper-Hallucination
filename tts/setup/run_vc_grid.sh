#!/usr/bin/env bash
# Run the VC / cloning grid: every candidate, in its own venv, over the same sources and
# targets. One system per call so a failure does not take the others down with it.
#
#   bash tts/setup/run_vc_grid.sh smoke      # 2 clips per system -- catches API drift cheap
#   bash tts/setup/run_vc_grid.sh full
#
# GPUs 6-7 only (CLAUDE.md); VC and cloning arms alternate between them.
set -uo pipefail

ROOT=/root/whisper-halluc-train
cd "$ROOT"
mode=${1:-smoke}
GPU_A=${2:-6}          # both default to the pair CLAUDE.md reserves; pass one twice to
GPU_B=${3:-7}          # keep the other free for a benchmark run
LIMIT=""
[ "$mode" = "smoke" ] && LIMIT="--limit 2"

run() {  # run <label> <venv> <gpu> <script> [extra args...]
  local label=$1 venv=$2 gpu=$3; shift 3
  echo "=== $label ($venv, cuda:$gpu) ==="
  "$ROOT/$venv/bin/python" "$@" --device "cuda:$gpu" $LIMIT 2>&1 | tail -25
  echo "=== $label exit ${PIPESTATUS[0]} ==="
}

set -a; . "$ROOT/.env"; set +a

run knnvc           .venv_knnvc     $GPU_A tts/vc_knnvc.py
run seedvc          .venv_seedvc    $GPU_A tts/vc_seedvc.py
run openvoice       .venv_openvoice $GPU_B tts/vc_openvoice.py --mode vc
run cosyvoice       .venv_cosyvoice $GPU_B tts/vc_cosyvoice.py --mode vc
run openvoice_clone .venv_openvoice $GPU_B tts/vc_openvoice.py --mode clone
run cosyvoice_clone .venv_cosyvoice $GPU_B tts/vc_cosyvoice.py --mode clone

echo "grid done: $(ls -d tts/vc_out/*/ 2>/dev/null | wc -l) systems under tts/vc_out"
