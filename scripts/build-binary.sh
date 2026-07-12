#!/usr/bin/env bash
# Path B: ``pulsing`` single binary (RustPython VM — pure Rust, no libpython).
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE="${1:-}"
if [[ "$RELEASE" == "--release" ]]; then
  cargo build --release -p pulsing-cli
  BIN=target/release/pulsing
else
  cargo build -p pulsing-cli
  BIN=target/debug/pulsing
fi

echo "==> Built $BIN (RustPython / rustpython_vm)"
