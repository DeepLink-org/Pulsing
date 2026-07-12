#!/usr/bin/env bash
# Sync Codex tool-related crates into vendor/codex-rs for pulsing-forge development.
# Source: https://github.com/openai/codex (Apache-2.0)
#
# Usage:
#   CODEX_ROOT=/path/to/codex/codex-rs ./scripts/sync-codex-forge.sh
#
# After sync, update docs/design/pulsing-forge.md "Synced from Codex" table.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODEX_ROOT="${CODEX_ROOT:-$(dirname "$ROOT")/codex/codex-rs}"
DEST="$ROOT/vendor/codex-rs"

if [[ ! -d "$CODEX_ROOT" ]]; then
  echo "error: Codex tree not found at $CODEX_ROOT" >&2
  echo "  clone codex or set CODEX_ROOT" >&2
  exit 1
fi

CRATES=(
  tools
  sandboxing
  execpolicy
  apply-patch
  file-system
  shell-command
  protocol
  utils/absolute-path
  utils/output-truncation
  utils/path-uri
  utils/string
  network-proxy
  bwrap
  linux-sandbox
)

mkdir -p "$DEST"
COMMIT="$(git -C "$(dirname "$CODEX_ROOT")" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Syncing from codex @ $COMMIT"

for rel in "${CRATES[@]}"; do
  src="$CODEX_ROOT/$rel"
  if [[ ! -d "$src" ]]; then
    echo "skip missing: $rel" >&2
    continue
  fi
  name="${rel//\//-}"
  mkdir -p "$DEST/$name"
  rsync -a --delete \
    --exclude 'target' \
    --exclude 'Cargo.lock' \
    "$src/" "$DEST/$name/"
  echo "  synced $rel -> vendor/codex-rs/$name"
done

cat > "$DEST/SYNC_INFO" <<EOF
codex_root=$CODEX_ROOT
commit=$COMMIT
synced_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
crates=${CRATES[*]}
EOF

echo "done -> $DEST (see SYNC_INFO)"
