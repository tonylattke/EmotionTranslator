#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> EmotionTranslator build"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required (e.g. sudo apt install ffmpeg)."
  exit 1
fi

PYTHON=""
for cand in python3.11 python3.10 python3.9 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    PYTHON="$cand"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "ERROR: python3 not found."
  exit 1
fi
PYVER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using Python $("$PYTHON" --version) (${PYVER})"

if [[ "$PYVER" == "3.12" ]] || [[ "$PYVER" == "3.13" ]]; then
  echo "WARNING: OpenVoice pins older packages; using relaxed dependency install for Python ${PYVER}."
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  "$PYTHON" -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

export PIP_DEFAULT_TIMEOUT=300

# PyPI CDN (files.pythonhosted.org) is blocked on some networks; cdproxy mirrors wheels.
if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  if curl -fsS --connect-timeout 5 "https://cdproxy.sportradar.online/pypi/simple/pip/" >/dev/null 2>&1; then
    export PIP_INDEX_URL="https://cdproxy.sportradar.online/pypi/simple/"
    export PIP_TRUSTED_HOST="cdproxy.sportradar.online"
    echo "Using pip index: ${PIP_INDEX_URL}"
  else
    echo "Using default PyPI index (set PIP_INDEX_URL if downloads fail)."
  fi
fi

pip_install_retry() {
  local tries=5
  local n=1
  until pip install "$@"; do
    if [[ $n -ge $tries ]]; then
      echo "ERROR: pip install failed after ${tries} attempts: $*"
      return 1
    fi
    echo "pip install failed (attempt ${n}/${tries}); retrying in 15s..."
    sleep 15
  n=$((n + 1))
  done
}

if [[ ! -d "$ROOT/vendor/OpenVoice" ]]; then
  echo "==> Cloning OpenVoice"
  git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git "$ROOT/vendor/OpenVoice"
fi

echo "==> Installing PyTorch (CUDA wheels, no PyPI deps during torch install)"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
if pip install torch torchaudio --no-deps --index-url "$TORCH_INDEX"; then
  :
elif pip install torch torchaudio --no-deps --index-url https://download.pytorch.org/whl/cpu; then
  echo "CUDA wheels unavailable; installed CPU PyTorch."
else
  pip_install_retry torch torchaudio --no-deps
fi

echo "==> PyTorch runtime dependencies"
pip_install_retry \
  filelock typing-extensions sympy networkx jinja2 fsspec setuptools

install_openvoice_deps() {
  if [[ "$PYVER" == "3.12" ]] || [[ "$PYVER" == "3.13" ]]; then
    echo "==> Installing OpenVoice dependencies (Python ${PYVER} compatible)"
    pip_install_retry \
      "librosa>=0.9.1" \
      "faster-whisper>=0.9.0" \
      "pydub>=0.25.1" \
      "wavmark>=0.0.3" \
      "numpy>=1.26.0" \
      "eng_to_ipa>=0.0.2" \
      "inflect>=7.0.0" \
      "unidecode>=1.3.7" \
      "whisper-timestamped>=1.14.2" \
      "pypinyin>=0.50.0" \
      "cn2an>=0.5.22" \
      "jieba>=0.42.1" \
      "gradio==3.48.0" \
      "langid>=1.1.6" \
      "python-dotenv>=1.0.0" \
      "soundfile>=0.12.0" \
      "av>=11.0.0" \
      "ctranslate2>=4.0.0"
  else
    pip_install_retry -r "$ROOT/vendor/OpenVoice/requirements.txt"
    pip_install_retry python-dotenv soundfile
  fi
}

install_openvoice_deps

echo "==> Installing OpenVoice package"
pip install -e "$ROOT/vendor/OpenVoice" --no-deps

pip_install_retry -r "$ROOT/requirements-app.txt"

echo "==> Pinning Gradio 3.48-compatible web stack"
pip_install_retry -r "$ROOT/requirements-gradio-stack.txt"

CKPT_ZIP="$ROOT/vendor/checkpoints_1226.zip"
CHECKPOINT_MARKER="$ROOT/vendor/OpenVoice/checkpoints/base_speakers/EN/checkpoint.pth"
CKPT_URLS=(
  "https://huggingface.co/camenduru/OpenVoice/resolve/main/checkpoints_1226.zip"
  "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip"
)

if [[ ! -f "$CHECKPOINT_MARKER" ]]; then
  echo "==> Downloading OpenVoice V1 checkpoints (~400MB)"
  rm -f "$CKPT_ZIP"
  downloaded=0
  for url in "${CKPT_URLS[@]}"; do
    echo "Trying $url"
    if curl -fL --retry 3 --retry-delay 5 "$url" -o "$CKPT_ZIP" && file "$CKPT_ZIP" | grep -qi zip; then
      downloaded=1
      break
    fi
    rm -f "$CKPT_ZIP"
  done
  if [[ "$downloaded" -ne 1 ]]; then
    echo "ERROR: checkpoint download failed. Place checkpoints under vendor/OpenVoice/checkpoints manually."
    exit 1
  fi
  unzip -q -o "$CKPT_ZIP" -d "$ROOT/vendor/OpenVoice"
  rm -f "$CKPT_ZIP"
fi

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
fi

echo "==> Verifying imports and checkpoints"
python -c "import torch; import openvoice; import gradio; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
test -f "$CHECKPOINT_MARKER"

echo "==> Build complete. Run: ./run.sh  (or ./run:.sh)"
