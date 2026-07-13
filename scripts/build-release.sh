#!/usr/bin/env bash
# Build both distribution paths: cross-platform wheels + single-binary CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RELEASE=0
MANYLINUX=0
WHEEL=1
BINARY=1
PACKAGE_BIN=0
OUT="dist"

usage() {
  cat <<'EOF'
Usage: scripts/build-release.sh [OPTIONS]

Build Python wheels and the pulsing single binary in one invocation.

Options:
  --release       Release builds (wheels + binary)
  --manylinux     Build manylinux_2_17 wheel (Linux CI / PyPI)
  --wheel-only    Skip pulsing-cli binary
  --binary-only   Skip maturin wheels
  --package       Create dist/pulsing-<platform>.tar.gz for the binary
  --out DIR       Wheel output directory (default: dist)
  -h, --help      Show this help

Examples:
  scripts/build-release.sh --release
  scripts/build-release.sh --release --manylinux --package
  just build-release
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) RELEASE=1; shift ;;
    --manylinux) MANYLINUX=1; shift ;;
    --wheel-only) BINARY=0; shift ;;
    --binary-only) WHEEL=0; shift ;;
    --package) PACKAGE_BIN=1; shift ;;
    --out)
      OUT="${2:?--out requires a directory}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ARGS=()
[[ "$RELEASE" -eq 1 ]] && ARGS+=(--release)
[[ "$MANYLINUX" -eq 1 ]] && ARGS+=(--manylinux)

echo "==> Pulsing release build (wheel=$WHEEL binary=$BINARY release=$RELEASE manylinux=$MANYLINUX)"

if [[ "$WHEEL" -eq 1 ]]; then
  "$ROOT/scripts/build-wheel.sh" "${ARGS[@]}" --out "$OUT"
fi

if [[ "$BINARY" -eq 1 ]]; then
  BIN_ARGS=()
  [[ "$RELEASE" -eq 1 ]] && BIN_ARGS+=(--release)
  [[ "$PACKAGE_BIN" -eq 1 ]] && BIN_ARGS+=(--package)
  [[ "$MANYLINUX" -eq 1 ]] && BIN_ARGS+=(--no-gui)
  "$ROOT/scripts/build-binary.sh" "${BIN_ARGS[@]}"
fi

echo ""
echo "==> Artifacts:"
[[ "$WHEEL" -eq 1 ]] && ls -lh "$OUT"/*.whl 2>/dev/null || true
[[ "$BINARY" -eq 1 ]] && ls -lh dist/bin/pulsing-* 2>/dev/null || true
[[ "$PACKAGE_BIN" -eq 1 ]] && ls -lh dist/pulsing-*.tar.gz dist/pulsing-*.zip 2>/dev/null || true
