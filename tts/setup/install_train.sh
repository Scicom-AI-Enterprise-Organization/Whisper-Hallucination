#!/bin/bash
# Training venv: transformers + peft for LoRA, separate from every inference venv because
# their torch pins conflict (CLAUDE.md).
set -x
cd /root/whisper-halluc-train
uv venv .venv_train --python 3.11
VIRTUAL_ENV=$PWD/.venv_train uv pip install -q torch transformers accelerate peft datasets \
    soundfile librosa numpy
.venv_train/bin/python -c "
import torch, transformers, peft
print('train venv ok | torch', torch.__version__, '| peft', peft.__version__,
      '| gpus', torch.cuda.device_count())
"
