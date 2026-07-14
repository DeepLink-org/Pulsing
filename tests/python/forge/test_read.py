# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Forge `Read` tool (Python fallback, `_read` in handlers.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.forge


def test_read_whole_file(local_forge, forge_workspace: Path) -> None:
    (forge_workspace / "a.txt").write_text("hello", encoding="utf-8")
    out = local_forge.call_tool("Read", {"file_path": "a.txt"})
    assert not out.is_error
    assert out.content == "hello"


def test_read_missing_file_reports_path(local_forge, forge_workspace: Path) -> None:
    out = local_forge.call_tool("Read", {"file_path": "missing.txt"})
    assert out.is_error
    assert "missing.txt" in out.content


def test_read_rejects_directory(local_forge, forge_workspace: Path) -> None:
    (forge_workspace / "sub").mkdir()
    out = local_forge.call_tool("Read", {"file_path": "sub"})
    assert out.is_error
    assert "directory" in out.content


def test_read_rejects_oversized_file(local_forge, forge_workspace: Path) -> None:
    (forge_workspace / "big.txt").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    out = local_forge.call_tool("Read", {"file_path": "big.txt"})
    assert out.is_error
    assert "too large" in out.content
    assert "offset" in out.content


def test_read_rejects_non_utf8(local_forge, forge_workspace: Path) -> None:
    (forge_workspace / "bin.dat").write_bytes(bytes([0xFF, 0xFE, 0x00, 0xFF]))
    out = local_forge.call_tool("Read", {"file_path": "bin.dat"})
    assert out.is_error
    assert "UTF-8" in out.content


def test_read_absolute_path_outside_cwd_is_allowed(
    local_forge, forge_workspace: Path
) -> None:
    """Read has no cwd boundary (unlike Write); it is a general-purpose reader."""
    outside = forge_workspace.parent / "outside.txt"
    outside.write_text("elsewhere", encoding="utf-8")
    try:
        out = local_forge.call_tool("Read", {"file_path": str(outside)})
        assert not out.is_error
        assert out.content == "elsewhere"
    finally:
        outside.unlink(missing_ok=True)


def test_read_offset_and_limit_page_through_lines(
    local_forge, forge_workspace: Path
) -> None:
    (forge_workspace / "lines.txt").write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
    out = local_forge.call_tool(
        "Read", {"file_path": "lines.txt", "offset": 2, "limit": 2}
    )
    assert not out.is_error
    assert out.content == "l2\nl3\n"


def test_read_offset_only_reads_to_end(local_forge, forge_workspace: Path) -> None:
    (forge_workspace / "lines.txt").write_text("l1\nl2\nl3\n", encoding="utf-8")
    out = local_forge.call_tool("Read", {"file_path": "lines.txt", "offset": 2})
    assert not out.is_error
    assert out.content == "l2\nl3\n"
