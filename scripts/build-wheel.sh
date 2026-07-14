#!/usr/bin/env bash
# Path A: Python wheel (extension-module). User's Python loads pulsing._core.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RELEASE=0
MANYLINUX=0
OUT="dist"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) RELEASE=1; shift ;;
    --manylinux) MANYLINUX=1; shift ;;
    --out)
      OUT="${2:?--out requires a directory}"
      shift 2
      ;;
    -h | --help)
      cat <<'EOF'
Usage: scripts/build-wheel.sh [OPTIONS]

Build pulsing Python wheels (extension-module via maturin).

Options:
  --release     Release build (default: develop / editable install)
  --manylinux   manylinux_2_17 wheel (for Linux CI / PyPI)
  --out DIR     Output directory (default: dist)
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUT"

if [[ "$MANYLINUX" -eq 1 ]]; then
  export CFLAGS="${CFLAGS:+$CFLAGS }-D_GNU_SOURCE"
fi

if [[ "$RELEASE" -eq 1 ]]; then
  MATURIN_ARGS=(build --release --out "$OUT")
  if [[ "$MANYLINUX" -eq 1 ]]; then
    MATURIN_ARGS+=(--compatibility manylinux_2_17 -i python3.10)
  fi
  maturin "${MATURIN_ARGS[@]}"
  maturin build --release --out "$OUT" --manifest-path crates/pulsing-bench-py/Cargo.toml
else
  maturin develop
  maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
fi

echo "==> Wheel path: ${OUT}/*.whl (extension-module)"
