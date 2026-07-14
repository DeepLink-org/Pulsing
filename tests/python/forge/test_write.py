# SPDX-License-Identifier: Apache-2.0
"""Write tool: new file, overwrite, path escape, deep parent dirs."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.testing.forge_harness import local_runtime

pytestmark = pytest.mark.forge


def test_write_creates_new_file(tmp_path: Path) -> None:
    rt = local_runtime(tmp_path)
    out = rt.call_tool("Write", {"file_path": "new.txt", "content": "hello"})
    assert not out.is_error
    assert out.content == "created"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello"


def test_write_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")
    rt = local_runtime(tmp_path)
    out = rt.call_tool("Write", {"file_path": "existing.txt", "content": "new"})
    assert not out.is_error
    assert out.content == "overwritten"
    assert target.read_text(encoding="utf-8") == "new"


def test_write_creates_deep_parent_dirs(tmp_path: Path) -> None:
    rt = local_runtime(tmp_path)
    out = rt.call_tool("Write", {"file_path": "a/b/c/deep.txt", "content": "deep"})
    assert not out.is_error
    assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text(
        encoding="utf-8"
    ) == "deep"


def test_write_rejects_relative_escape_outside_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rt = local_runtime(workspace)
    out = rt.call_tool("Write", {"file_path": "../escape.txt", "content": "x"})
    assert out.is_error
    assert "outside working directory" in out.content
    assert not (tmp_path / "escape.txt").exists()


def test_write_rejects_absolute_path_outside_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    rt = local_runtime(workspace)
    out = rt.call_tool("Write", {"file_path": str(outside), "content": "x"})
    assert out.is_error
    assert "outside working directory" in out.content
    assert not outside.exists()


def test_write_allows_absolute_path_inside_cwd(tmp_path: Path) -> None:
    rt = local_runtime(tmp_path)
    target = tmp_path / "abs.txt"
    out = rt.call_tool("Write", {"file_path": str(target), "content": "ok"})
    assert not out.is_error
    assert target.read_text(encoding="utf-8") == "ok"


def test_write_rejects_symlink_escape_outside_cwd(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "link").symlink_to(outside)
    rt = local_runtime(workspace)
    out = rt.call_tool("Write", {"file_path": "link/escape.txt", "content": "x"})
    assert out.is_error
    assert "outside working directory" in out.content
    assert not (outside / "escape.txt").exists()
