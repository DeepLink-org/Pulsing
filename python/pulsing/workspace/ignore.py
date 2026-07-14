# SPDX-License-Identifier: Apache-2.0
"""Default ignore rules for workspace checkpoints."""

from __future__ import annotations

from pathlib import Path

_SKIP_DIRS = {
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
}


def should_skip(rel: Path) -> bool:
    s = rel.as_posix()
    if not s or s.startswith(".pulsing/history"):
        return True
    if s == ".git" or s.startswith(".git/"):
        return True
    return any(part in _SKIP_DIRS for part in rel.parts)
