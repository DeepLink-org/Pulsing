#!/usr/bin/env bash
# Test the built wheel with the requested interpreter, never a uv tool's Python.
set -euo pipefail

PYTHON="${1:-python3}"
VENV="${PULSING_CI_VENV:-.venv-ci}"

if command -v uv >/dev/null 2>&1; then
  uv venv --clear --python "$PYTHON" "$VENV"
  uv pip install --python "$VENV/bin/python" dist/pulsing-*.whl pytest pytest-asyncio
else
  if [[ "$PYTHON" =~ ^[0-9]+\.[0-9]+$ ]]; then
    PYTHON="python$PYTHON"
  fi
  "$PYTHON" -m venv --clear "$VENV"
  "$VENV/bin/python" -m pip install dist/pulsing-*.whl pytest pytest-asyncio
fi

# set -e propagates both dependency installation and pytest failures.
"$VENV/bin/python" -m pytest tests/python -v
