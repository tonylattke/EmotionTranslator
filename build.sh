#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON=""
for cand in python3.11 python3.10 python3.12 python3.9 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)'; then
    PYTHON="$cand"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "ERROR: Python 3.9-3.12 not found (e.g. sudo apt install python3.11 python3.11-venv)."
  exit 1
fi

exec "$PYTHON" "$ROOT/scripts/setup_env.py" "$@"
