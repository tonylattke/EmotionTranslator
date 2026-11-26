#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Virtualenv missing. Run ./build.sh first."
  exit 1
fi

exec "$ROOT/.venv/bin/python" -X utf8 -m app.gradio_app "$@"
