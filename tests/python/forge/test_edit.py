# SPDX-License-Identifier: Apache-2.0
"""Edit tool: unique replace, ambiguity, path escape."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.testing.forge_harness import local_runtime

pytestmark = pytest.mark.forge


def test_edit_replaces_unique_occurrence(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")
    rt = local_runtime(tmp_path)
    out = rt.call_tool(
        "Edit",
        {"file_path": "a.txt", "old_string": "world", "new_string": "there"},
    )
    assert not out.is_error
    assert target.read_text(encoding="utf-8") == "hello there"


def test_edit_rejects_missing_old_string(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")
    rt = local_runtime(tmp_path)
    out = rt.call_tool(
        "Edit",
        {"file_path": "a.txt", "old_string": "nope", "new_string": "x"},
    )
    assert out.is_error
    assert "not found" in out.content
    assert target.read_text(encoding="utf-8") == "hello world"


def test_edit_rejects_ambiguous_old_string_with_count(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("a a a", encoding="utf-8")
    rt = local_runtime(tmp_path)
    out = rt.call_tool(
        "Edit",
        {"file_path": "a.txt", "old_string": "a", "new_string": "b"},
    )
    assert out.is_error
    assert "3 occurrences" in out.content
    assert target.read_text(encoding="utf-8") == "a a a"


def test_edit_rejects_relative_escape_outside_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "a.txt"
    target.write_text("hello world", encoding="utf-8")
    rt = local_runtime(workspace)
    out = rt.call_tool(
        "Edit",
        {"file_path": "../a.txt", "old_string": "world", "new_string": "there"},
    )
    assert out.is_error
    assert "outside working directory" in out.content
    assert target.read_text(encoding="utf-8") == "hello world"


def test_edit_rejects_absolute_path_outside_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    rt = local_runtime(workspace)
    out = rt.call_tool(
        "Edit",
        {"file_path": str(outside), "old_string": "secret", "new_string": "leaked"},
    )
    assert out.is_error
    assert "outside working directory" in out.content
    assert outside.read_text(encoding="utf-8") == "secret"


def test_edit_allows_absolute_path_inside_cwd(tmp_path: Path) -> None:
    target = tmp_path / "inside.txt"
    target.write_text("hello world", encoding="utf-8")
    rt = local_runtime(tmp_path)
    out = rt.call_tool(
        "Edit",
        {
            "file_path": str(target),
            "old_string": "world",
            "new_string": "there",
        },
    )
    assert not out.is_error
    assert target.read_text(encoding="utf-8") == "hello there"


def test_edit_rejects_missing_file(tmp_path: Path) -> None:
    rt = local_runtime(tmp_path)
    out = rt.call_tool(
        "Edit",
        {"file_path": "missing.txt", "old_string": "a", "new_string": "b"},
    )
    assert out.is_error
    assert "file not found" in out.content


def test_edit_rejects_directory_path(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    rt = local_runtime(tmp_path)
    out = rt.call_tool(
        "Edit",
        {"file_path": "sub", "old_string": "a", "new_string": "b"},
    )
    assert out.is_error
    assert "not a file" in out.content


def test_edit_rejects_symlink_escape_outside_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    link.symlink_to(outside)
    rt = local_runtime(workspace)
    out = rt.call_tool(
        "Edit",
        {"file_path": "link.txt", "old_string": "secret", "new_string": "leaked"},
    )
    assert out.is_error
    assert "outside working directory" in out.content
    assert outside.read_text(encoding="utf-8") == "secret"
    assert link.is_symlink()
