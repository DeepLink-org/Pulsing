# SPDX-License-Identifier: Apache-2.0
"""Glob tool unit tests — matching, missing path, pattern safety, truncation."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.forge


def _glob(rt, pattern: str, path: str | None = None):
    args: dict[str, str] = {"pattern": pattern}
    if path is not None:
        args["path"] = path
    return rt.call_tool("Glob", args)


def test_glob_finds_matching_files(local_forge, forge_workspace: Path) -> None:
    (forge_workspace / "a.txt").write_text("x", encoding="utf-8")
    (forge_workspace / "b.rs").write_text("x", encoding="utf-8")
    out = _glob(local_forge, "*.txt", str(forge_workspace))
    assert not out.is_error
    assert out.content.endswith("a.txt")


def test_glob_reports_no_matches(local_forge, forge_workspace: Path) -> None:
    out = _glob(local_forge, "*.nope", str(forge_workspace))
    assert not out.is_error
    assert out.content == "(no matches)"


def test_glob_rejects_missing_path(local_forge, forge_workspace: Path) -> None:
    out = _glob(local_forge, "*", str(forge_workspace / "does" / "not" / "exist"))
    assert out.is_error
    assert "does not exist" in out.content


def test_glob_rejects_absolute_pattern(local_forge, forge_workspace: Path) -> None:
    # An absolute pattern must not be able to escape `path`/cwd.
    out = _glob(local_forge, "/etc/*", str(forge_workspace))
    assert out.is_error
    assert "absolute" in out.content


def test_glob_truncates_with_clear_message(local_forge, forge_workspace: Path) -> None:
    for i in range(510):
        (forge_workspace / f"f{i}.txt").write_text("x", encoding="utf-8")
    out = _glob(local_forge, "*.txt", str(forge_workspace))
    assert not out.is_error
    assert "truncated: showing 500 of 510 matches" in out.content
