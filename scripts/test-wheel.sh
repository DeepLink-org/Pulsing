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

# Child processes must resolve python/python3 from the selected matrix environment,
# including on images that have only uv-managed interpreters and no system Python.
VIRTUAL_ENV="$(cd "$VENV" && pwd)"
export VIRTUAL_ENV
export PATH="$VIRTUAL_ENV/bin:$PATH"

# set -e propagates both dependency installation and pytest failures.
"$VIRTUAL_ENV/bin/python" -m pytest tests/python -v
