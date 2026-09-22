#!/bin/bash
set -x
cd /root/whisper-halluc-train
uv venv .venv_mms --python 3.11 2>&1 | tail -2
VIRTUAL_ENV=$PWD/.venv_mms uv pip install -q torch transformers soundfile librosa numpy 2>&1 | tail -3
.venv_mms/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__, 'cuda', torch.cuda.is_available())"
