# SPDX-License-Identifier: Apache-2.0
"""Workspace paths and discovery."""

from __future__ import annotations

import hashlib
from pathlib import Path

PULSING_DIR = ".pulsing"
WORKSPACE_FILE = "workspace.json"
CLUSTER_FILE = "cluster.json"
HISTORY_DIR = "history"
REVISIONS_DIR = "revisions"
HEAD_FILE = "HEAD"
HOOKS_DIR = "hooks"
SCRIPTS_DIR = "scripts"


def workspace_cluster_id(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]


def find_workspace_root(start: Path | None = None) -> Path | None:
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
            "not a Pulsing workspace — run `pulsing init` in this project directory first",
        )
    return root


class WorkspaceLayout:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def pulsing_dir(self) -> Path:
        return self.root / PULSING_DIR

    @property
    def workspace_file(self) -> Path:
        return self.pulsing_dir / WORKSPACE_FILE

    @property
    def cluster_file(self) -> Path:
        return self.pulsing_dir / CLUSTER_FILE

    @property
    def hooks_dir(self) -> Path:
        return self.pulsing_dir / HOOKS_DIR

    @property
    def scripts_dir(self) -> Path:
        return self.pulsing_dir / SCRIPTS_DIR

    @property
    def history_dir(self) -> Path:
        return self.pulsing_dir / HISTORY_DIR

    @property
    def revisions_dir(self) -> Path:
        return self.history_dir / REVISIONS_DIR

    @property
    def head_file(self) -> Path:
        return self.history_dir / HEAD_FILE

    def revision_dir(self, revision_id: str) -> Path:
        return self.revisions_dir / revision_id

    def is_initialized(self) -> bool:
        return self.cluster_file.is_file()
