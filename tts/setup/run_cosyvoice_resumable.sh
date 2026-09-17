#!/usr/bin/env bash
# CosyVoice2 crashes the flow encoder with SIGFPE (a signal -- Python cannot catch it) when
# the source is longer than the speaker prompt. With REF=long that mostly does not happen;
# with the 6 s reference it happens constantly. Restart until the manifest is complete:
# --resume skips finished pairs, --chunk exits cleanly before a long run destabilises, and a
# pair that crashes twice is written off rather than retried forever.
#
#   REF=long  CHUNK=25 bash tts/setup/run_cosyvoice_resumable.sh vc 7
#   REF=short CHUNK=10 bash tts/setup/run_cosyvoice_resumable.sh clone 7
set -uo pipefail

ROOT=/root/whisper-halluc-train
cd "$ROOT"
mode=${1:-vc}
gpu=${2:-7}
want=${3:-204}
REF=${REF:-long}
CHUNK=${CHUNK:-25}

# Output dir mirrors the naming the other candidates use, so the scorer sees one system each.
if [ "$mode" = clone ]; then sys=cosyvoice_clone
elif [ "$REF" = long ]; then sys=cosyvoice_longref
else sys=cosyvoice; fi
out="$ROOT/tts/vc_out"

set -a; . "$ROOT/.env"; set +a
# Deliberately NOT pinning OMP_NUM_THREADS: the only configuration observed to complete 80
# conversions in one process had the default thread pool. Pinning it to 1 did not help and
# may have hurt.

for attempt in $(seq 1 120); do
  rows=$(( $(wc -l < "$out/$sys/manifest.csv" 2>/dev/null || echo 1) - 1 ))
  [ "$rows" -ge "$want" ] && { echo "[$sys] complete: $rows rows"; exit 0; }
  echo "--- $sys attempt $attempt (have $rows/$want) ---"
  "$ROOT/.venv_cosyvoice/bin/python" tts/vc_cosyvoice.py --mode "$mode" --device "cuda:$gpu" \
      --ref "$REF" --out "$out" --system-name "$sys" --resume --chunk "$CHUNK" 2>&1 \
      | grep -vE "it/s\]|DEBUG|INFO|WARNING" | tail -3
done
echo "[$sys] gave up after 120 attempts"
