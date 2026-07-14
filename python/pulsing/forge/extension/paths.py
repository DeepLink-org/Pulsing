# SPDX-License-Identifier: Apache-2.0
"""Filesystem roots for Extension tools (aligned with Codex layout)."""

from __future__ import annotations

import os
from pathlib import Path

from pulsing.forge.discovery.codex_paths import codex_home


def agents_skills_roots(cwd: Path) -> list[Path]:
    roots: list[Path] = []
    user = Path.home() / ".agents" / "skills"
    if user.is_dir():
        roots.append(user.resolve())
    repo = (cwd / ".agents" / "skills").resolve()
    if repo.is_dir():
        roots.append(repo)
    extra = os.environ.get("FORGE_SKILLS_DIRS", "").strip()
    for part in extra.split(os.pathsep):
        p = Path(part).expanduser()
        if p.is_dir():
            roots.append(p.resolve())
    return roots


def memories_root() -> Path:
    raw = os.environ.get("FORGE_MEMORIES_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (codex_home() / "memories").resolve()
