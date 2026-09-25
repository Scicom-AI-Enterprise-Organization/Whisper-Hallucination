#!/bin/bash
# faster-whisper (CTranslate2) for the ablation grid: repetition_penalty and
# no_repeat_ngram_size exist there and nowhere else in the Whisper family.
set -x
cd /root/whisper-halluc-train
uv venv .venv_ablate --python 3.11
VIRTUAL_ENV=$PWD/.venv_ablate uv pip install -q faster-whisper datasets soundfile librosa numpy
.venv_ablate/bin/python -c "
from faster_whisper import WhisperModel
import datasets, soundfile
print('ablate venv ok')
"
