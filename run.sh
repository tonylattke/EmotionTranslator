#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Virtualenv missing. Run ./build.sh first."
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
# Avoid corporate proxies breaking Gradio's localhost health check.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

exec python -m app.gradio_app "$@"
