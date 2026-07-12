# SPDX-License-Identifier: Apache-2.0
"""Find workspace root (the world) via ``.pulsing/cluster.json``."""

from __future__ import annotations

import hashlib
from pathlib import Path

PULSING_DIR = ".pulsing"
CLUSTER_FILE = "cluster.json"
NODE_FILE = "node.json"


def workspace_cluster_id(root: Path) -> str:
    """Stable 12-char id from absolute workspace path."""
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]


def find_workspace_root(start: Path | None = None) -> Path | None:
    """Return directory containing ``.pulsing/cluster.json``, or None."""
    cur = (start or Path.cwd()).resolve()
    while True:
        if (cur / PULSING_DIR / CLUSTER_FILE).is_file():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def require_workspace_root(start: Path | None = None) -> Path:
    root = find_workspace_root(start)
    if root is None:
        raise SystemExit(
            "not a workspace yet — run `pulsing init` in this project directory first",
        )
    return root
