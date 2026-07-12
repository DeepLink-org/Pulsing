#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Bootstrap a minimal Pulsing workspace demo from zero (no API key by default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO_DIR="${PULSING_DEMO_DIR:-$(mktemp -d -t pulsing-ws-demo)}"
KEEP=0

usage() {
  cat <<'EOF'
usage: workspace_demo.sh [options] [-- message]

Bootstrap .pulsing/, wake guide with demo LLM, send one message.

Options:
  --dir PATH       workspace directory (default: temp dir)
  --keep           keep workspace directory after run
  --provider NAME  demo | anthropic | openai (default: demo)
  --real-llm       alias for --provider anthropic
  -h, --help       show this help

Environment:
  PULSING_DEMO_DIR  default workspace directory when --dir is omitted

Examples:
  ./examples/python/workspace_demo.sh
  ./examples/python/workspace_demo.sh --dir ./my-ws --keep
  ./examples/python/workspace_demo.sh -- "read README and summarize"
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      DEMO_DIR="$2"
      shift 2
      ;;
    --keep)
      KEEP=1
      shift
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --real-llm)
      PROVIDER="anthropic"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

PROVIDER="${PROVIDER:-demo}"
PY_ARGS=(--dir "$DEMO_DIR" --provider "$PROVIDER")
[[ "$KEEP" -eq 1 ]] && PY_ARGS+=(--keep)
[[ ${#ARGS[@]} -gt 0 ]] && PY_ARGS+=(--message "${ARGS[*]}")

cd "$ROOT"
if command -v uv >/dev/null 2>&1; then
  exec uv run python examples/python/workspace_minimal_demo.py "${PY_ARGS[@]}"
fi
exec python examples/python/workspace_minimal_demo.py "${PY_ARGS[@]}"
