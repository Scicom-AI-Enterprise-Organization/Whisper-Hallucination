#!/bin/bash
# One uv venv per model -- their torch / transformers pins do not coexist.
#   bash tts/setup/install_tts_extra.sh xtts|chatterbox|voxcpm
set -x
cd /root/whisper-halluc-train
sys=$1
case "$sys" in
  xtts)
    uv venv .venv_xtts --python 3.11
    # coqui-tts imports `isin_mps_friendly`, removed in transformers 5, and from torch 2.9
    # it does audio IO through torchcodec. Both are hard requirements, not preferences.
    VIRTUAL_ENV=$PWD/.venv_xtts uv pip install -q coqui-tts "transformers<5" torch torchaudio \
        torchcodec soundfile librosa
    ;;
  chatterbox)
    uv venv .venv_chatterbox --python 3.11
    # chatterbox constructs a perth watermarker unconditionally; without the package it dies
    # with "TypeError: 'NoneType' object is not callable".
    VIRTUAL_ENV=$PWD/.venv_chatterbox uv pip install -q chatterbox-tts resemble-perth \
        soundfile librosa
    ;;
  voxcpm)
    uv venv .venv_voxcpm --python 3.11
    VIRTUAL_ENV=$PWD/.venv_voxcpm uv pip install -q voxcpm soundfile librosa
    ;;
  kokoro)
    uv venv .venv_kokoro --python 3.11
    VIRTUAL_ENV=$PWD/.venv_kokoro uv pip install -q "kokoro>=0.9.2" soundfile librosa
    ;;
  qwen3tts)
    uv venv .venv_qwen3tts --python 3.11
    VIRTUAL_ENV=$PWD/.venv_qwen3tts uv pip install -q qwen-tts soundfile librosa
    ;;
  moss)
    uv venv .venv_moss --python 3.11
    # its processor loads the reference clip through load_with_torchcodec
    VIRTUAL_ENV=$PWD/.venv_moss uv pip install -q torch torchaudio transformers accelerate \
        torchcodec soundfile librosa
    ;;
  *)
    echo "unknown system: $sys"; exit 1
    ;;
esac
