#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash ./start_wsl.sh [port] [extra serve args...]

Start the translator web console inside WSL using the local `.venv` virtualenv.
Examples:
  bash ./start_wsl.sh
  bash ./start_wsl.sh 10002
  bash ./start_wsl.sh 10002 --workers 4
EOF
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1:-10001}"
if [[ $# -gt 0 ]]; then
  shift
fi

HOST="${TRANSLATOR_HOST:-127.0.0.1}"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"

exec "$PYTHON_BIN" -m translator serve --host "$HOST" --port "$PORT" "$@"
