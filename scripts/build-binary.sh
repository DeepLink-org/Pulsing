#!/usr/bin/env bash
# Path B: ``pulsing`` single binary (RustPython VM — pure Rust, no libpython).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/lib/platform.sh
source "$ROOT/scripts/lib/platform.sh"

RELEASE=0
OUT="dist/bin"
PACKAGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) RELEASE=1; shift ;;
    --out)
      OUT="${2:?--out requires a directory}"
      shift 2
      ;;
    --package) PACKAGE=1; shift ;;
    -h | --help)
      cat <<'EOF'
Usage: scripts/build-binary.sh [OPTIONS]

Build the pulsing single-binary CLI (pulsing-cli / RustPython).

Options:
  --release   Release build (default: debug)
  --out DIR   Directory for the binary (default: dist/bin)
  --package   Also create dist/pulsing-<platform>.tar.gz (or .zip on Windows)
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

TAG="$(platform_tag)"
mkdir -p "$OUT"

if [[ "$RELEASE" -eq 1 ]]; then
  cargo build --release -p pulsing-cli
  SRC="target/release/pulsing"
else
  cargo build -p pulsing-cli
  SRC="target/debug/pulsing"
fi

DEST="$OUT/pulsing-${TAG}"
cp "$SRC" "$DEST"
chmod +x "$DEST"

echo "==> Binary: $DEST"

if [[ "$PACKAGE" -eq 1 ]]; then
  PKG_DIR="$ROOT/dist"
  mkdir -p "$PKG_DIR"
  case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN* | Windows*)
      ARCHIVE="$PKG_DIR/pulsing-${TAG}.zip"
      (cd "$OUT" && zip -q "$ARCHIVE" "pulsing-${TAG}")
      ;;
    *)
      ARCHIVE="$PKG_DIR/pulsing-${TAG}.tar.gz"
      tar -czf "$ARCHIVE" -C "$OUT" "pulsing-${TAG}"
      ;;
  esac
  echo "==> Archive: $ARCHIVE"
fi
