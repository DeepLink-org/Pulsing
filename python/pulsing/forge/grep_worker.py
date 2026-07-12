# SPDX-License-Identifier: Apache-2.0
"""Spawn-safe grep scan worker (stdlib only — no pulsing.forge package import)."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

GREP_MAX = 200


def scan(
    root: Path,
    pattern: str,
    glob: str | None,
    boundary_s: str | None,
) -> tuple[list[str], int]:
    boundary = Path(boundary_s) if boundary_s else None
    cre = re.compile(pattern)
    hits: list[str] = []
    total = 0

    def within_boundary(fp: Path) -> bool:
        if boundary is None:
            return True
        try:
            fp.resolve().relative_to(boundary.resolve())
        except ValueError:
            return False
        return True

    def consider_file(fp: Path) -> None:
        nonlocal total
        if not within_boundary(fp):
            return
        if glob and not fnmatch.fnmatch(fp.name, glob):
            return
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        for i, line in enumerate(text.splitlines(), 1):
            if cre.search(line):
                total += 1
                if len(hits) < GREP_MAX:
                    hits.append(f"{fp}:{i}:{line[:500]}")

    if root.is_file():
        consider_file(root)
    else:
        for fp in root.rglob("*"):
            if fp.is_file():
                consider_file(fp)
    return hits, total


def worker(
    root_s: str,
    pattern: str,
    glob: str | None,
    boundary_s: str | None,
    q: Any,
) -> None:
    try:
        hits, total = scan(Path(root_s), pattern, glob, boundary_s)
        q.put(("ok", hits, total))
    except Exception as exc:
        q.put(("err", str(exc)))
