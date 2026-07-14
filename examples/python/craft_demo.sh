#!/usr/bin/env bash
# One-shot agent workspace demo: three chattering agents + Zellij dashboard (if installed).
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
shift || true
if command -v pulsing-agent >/dev/null 2>&1; then
  exec pulsing-agent demo "$@"
fi
exec pulsing agent demo "$@"
