#!/bin/bash
# Re-run any (model, collection-set) baseline whose jsonl is missing, whatever the cause.
# Written as a loop over what is ON DISK rather than a hand-kept list, because three separate
# runs have now been lost to `sync --delete` and re-sequencing them by hand kept missing one.
cd /root/whisper-halluc-train
MODELS="openai/whisper-large-v2 openai/whisper-large-v3 openai/whisper-large-v3-turbo mesolitica/malaysian-whisper-large-v2 mesolitica/Malaysian-whisper-large-v3-turbo-v3"
run_set () {              # $1 = out dir, rest = roots
  out=$1; shift
  for m in $MODELS; do
    tag=$(basename $m)
    if [ -s "$out/$tag.summary.json" ]; then
      echo "have   $out/$tag"
      continue
    fi
    echo "=== run $out/$tag ==="
    stdbuf -oL .venv_wild/bin/python bench/run_wild_baseline.py --model $m \
        --roots "$@" --out "$out" --device cuda:7 --batch-size 24 2>&1 \
        | grep -E "blank_speech|aphasia|halas_|loop |Traceback" | tail -4
  done
}
run_set bench/wild_results        audio_wild/halas
run_set bench/wild_results_noisy  audio_wild/audioset audio_wild/gigaspeech_xs audio_wild/peoples_dirty
run_set bench/wild_results_mined  audio_wild/ami audio_wild/earnings22 audio_wild/peoples_speech audio_wild/voxpopuli audio/aphasia_koenecke
