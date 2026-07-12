#!/usr/bin/env bash
# Path A: Python wheel (extension-module). User's Python loads pulsing._core.so.
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE="${1:-}"
if [[ "$RELEASE" == "--release" ]]; then
  maturin build --release
  maturin build --release --manifest-path crates/pulsing-bench-py/Cargo.toml
else
  maturin develop
  maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
fi

echo "==> Wheel path (extension-module): pulsing._core cdylib + python/pulsing"
