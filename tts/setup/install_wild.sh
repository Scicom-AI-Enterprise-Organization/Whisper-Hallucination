#!/bin/bash
# Dedicated venv for wild mining. NOT .venv_bench: installing silero-vad there exposes the
# torchao/torch clash that venv already carries for neucodec ("cannot import name
# 'ScalingType' from torch.nn.functional"), and transformers imports torchao when present.
set -x
cd /root/whisper-halluc-train
uv venv .venv_wild --python 3.11
VIRTUAL_ENV=$PWD/.venv_wild uv pip install -q torch transformers datasets soundfile librosa \
    silero-vad onnxruntime numpy
.venv_wild/bin/python -c "
import torch, transformers, datasets
from silero_vad import load_silero_vad, get_speech_timestamps
load_silero_vad(onnx=True)
print('wild venv ok | torch', torch.__version__, '| cuda', torch.cuda.is_available())
"
