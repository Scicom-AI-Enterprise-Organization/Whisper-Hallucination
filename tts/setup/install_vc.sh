#!/usr/bin/env bash
# Install one voice-conversion / voice-cloning candidate into its OWN venv.
#
# Same reason as the TTS candidates (CLAUDE.md, "Model-specific setup"): these repos pin
# mutually incompatible torch versions -- seed-vc wants 2.4.0, CosyVoice 2.3.1, and the
# benchmark venv is on 2.14. One venv each, no shared site-packages.
#
#   bash tts/setup/install_vc.sh knnvc|seedvc|openvoice|cosyvoice
#
# Deliberately NOT installing: gradio / FreeSimpleGUI / sounddevice (demo UIs), deepspeed
# and tensorrt (training + serving), modelscope (duplicate model source). Inference only.
set -euo pipefail

ROOT=/root/whisper-halluc-train
REPOS=$ROOT/vc_repos
mkdir -p "$REPOS"
# NB: no braces in the :? message -- bash ends the expansion at the first "}".
sys=${1:?usage: install_vc.sh knnvc/seedvc/openvoice/cosyvoice}

venv_for() { echo "$ROOT/.venv_$1"; }
pip_in() { local v=$1; shift; VIRTUAL_ENV=$v uv pip install --python "$v/bin/python" "$@"; }

case "$sys" in

knnvc)
  # WavLM encoder + HiFi-GAN vocoder, both pulled by torch.hub; the repo vendors its own
  # WavLM copy, so there is nothing to pip-install beyond the three declared deps.
  v=$(venv_for knnvc)
  uv venv --python 3.12 "$v"
  pip_in "$v" torch torchaudio numpy soundfile
  [ -d "$REPOS/knn-vc" ] || git clone --depth 1 https://github.com/bshall/knn-vc "$REPOS/knn-vc"
  ;;

seedvc)
  # requirements.txt starts with four nightly-index lines that fight the pinned versions
  # below them; drop those and keep the pins.
  v=$(venv_for seedvc)
  [ -d "$REPOS/seed-vc" ] || git clone --depth 1 https://github.com/Plachtaa/seed-vc "$REPOS/seed-vc"
  grep -vE '^(torch|torchvision|torchaudio) --pre|^accelerate$|gradio|FreeSimpleGUI|sounddevice|modelscope' \
      "$REPOS/seed-vc/requirements.txt" > "$REPOS/seed-vc/requirements.inference.txt"
  uv venv --python 3.11 "$v"
  pip_in "$v" --index-strategy unsafe-best-match -r "$REPOS/seed-vc/requirements.inference.txt"
  pip_in "$v" accelerate
  ;;

openvoice)
  # Only the ToneColorConverter is needed (v2 keeps the base TTS in MeloTTS, installed
  # separately for the cloning arm). numpy==1.22 / librosa==0.9.1 in their requirements are
  # demo-era pins with no wheels for modern Python -- take the modern pair instead.
  v=$(venv_for openvoice)
  [ -d "$REPOS/OpenVoice" ] || git clone --depth 1 https://github.com/myshell-ai/OpenVoice "$REPOS/OpenVoice"
  uv venv --python 3.11 "$v"
  pip_in "$v" torch torchaudio "numpy<2" "librosa>=0.10" soundfile pydub inflect unidecode \
              eng_to_ipa pypinyin cn2an jieba langid whisper-timestamped wavmark
  pip_in "$v" "huggingface_hub>=0.34"
  ;;

cosyvoice)
  # CosyVoice2 needs its Matcha-TTS submodule on sys.path (the repo does this in its own
  # examples too). deepspeed / tensorrt are training + TRT-serving only.
  v=$(venv_for cosyvoice)
  [ -d "$REPOS/CosyVoice" ] || git clone --recursive --depth 1 https://github.com/FunAudioLLM/CosyVoice "$REPOS/CosyVoice"
  # openai-whisper==20231117 has no pyproject and its setup.py imports pkg_resources, which
  # uv's build isolation does not provide -- the 2024 release builds cleanly and CosyVoice
  # only uses it for the text frontend.
  grep -vE 'deepspeed|tensorrt|gradio|fastapi|uvicorn|grpcio|tensorboard|gdown|modelscope|onnxruntime-gpu|openai-whisper|extra-index-url' \
      "$REPOS/CosyVoice/requirements.txt" > "$REPOS/CosyVoice/requirements.inference.txt"
  echo "onnxruntime==1.18.0" >> "$REPOS/CosyVoice/requirements.inference.txt"
  uv venv --python 3.10 "$v"
  pip_in "$v" --index-strategy unsafe-best-match -r "$REPOS/CosyVoice/requirements.inference.txt"
  # openai-whisper's setup.py imports pkg_resources, which setuptools REMOVED in 81. So it
  # needs both a build env that has setuptools (--no-build-isolation) and one old enough to
  # still ship pkg_resources. CosyVoice's frontend imports `whisper` at module level, so
  # this is not optional.
  pip_in "$v" "setuptools<81" wheel
  VIRTUAL_ENV=$v uv pip install --python "$v/bin/python" --no-build-isolation \
      openai-whisper==20240930 || echo "[warn] openai-whisper skipped -- import path untested"
  ;;

*) echo "unknown system: $sys" >&2; exit 2 ;;
esac

echo "[ok] $sys installed -> $(venv_for "$sys")"
