# SPDX-License-Identifier: Apache-2.0
"""Grep tool — behavior, safety, and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.handlers import _grep

pytestmark = [pytest.mark.forge, pytest.mark.forge_l2]


def _ctx(tmp_path: Path) -> ToolCallContext:
    return ToolCallContext(cwd=tmp_path)


def test_grep_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\nworld\nhello again\n", encoding="utf-8")
    out = _grep(ctx=_ctx(tmp_path), pattern="hello", path=str(tmp_path))
    assert not out.is_error
    assert out.content.count("hello") == 2
    assert "a.txt:1:hello" in out.content


def test_grep_invalid_regex(tmp_path: Path) -> None:
    out = _grep(ctx=_ctx(tmp_path), pattern="(unclosed", path=str(tmp_path))
    assert out.is_error
    assert "Invalid regex" in out.content


def test_grep_path_not_found(tmp_path: Path) -> None:
    out = _grep(ctx=_ctx(tmp_path), pattern="x", path=str(tmp_path / "missing"))
    assert out.is_error
    assert "path not found" in out.content


def test_grep_path_outside_cwd_is_allowed(tmp_path: Path) -> None:
    """Grep (like Glob) does not sandbox path — document current behavior."""
    outside = tmp_path.parent / "outside_grep.txt"
    outside.write_text("secret_token\n", encoding="utf-8")
    try:
        out = _grep(ctx=_ctx(tmp_path), pattern="secret_token", path=str(outside))
        assert not out.is_error
        assert "secret_token" in out.content
    finally:
        outside.unlink(missing_ok=True)


def test_grep_redos_pattern_returns_within_timeout(tmp_path: Path) -> None:
    """Pathological pattern must not block the tool call indefinitely."""
    evil_line = "a" * 40 + "!"
    (tmp_path / "evil.txt").write_text(evil_line, encoding="utf-8")
    out = _grep(ctx=_ctx(tmp_path), pattern="(a+)+b", path=str(tmp_path))
    # Either no match or explicit timeout error — must not hang.
    assert out.is_error or out.content == "(no matches)" or "timed out" in out.content


def test_grep_pattern_too_long(tmp_path: Path) -> None:
    out = _grep(ctx=_ctx(tmp_path), pattern="a" * 1001, path=str(tmp_path))
    assert out.is_error
    assert "Pattern too long" in out.content


def test_grep_truncation_message(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("hit\n" * 250, encoding="utf-8")
    out = _grep(ctx=_ctx(tmp_path), pattern="hit", path=str(tmp_path))
    assert not out.is_error
    assert "truncated: showing 200 of 250 matches" in out.content


def test_grep_relative_path_resolves_against_ctx_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("needle\n", encoding="utf-8")
    other = tmp_path.parent / "other_cwd"
    other.mkdir(exist_ok=True)
    monkeypatch.chdir(other)
    out = _grep(ctx=_ctx(tmp_path), pattern="needle", path="sub")
    assert not out.is_error
    assert "needle" in out.content


def test_grep_skips_symlink_outside_cwd(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_grep_secret.txt"
    outside.write_text("outside_secret\n", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    try:
        out = _grep(ctx=_ctx(tmp_path), pattern="outside_secret", path=str(tmp_path))
        assert not out.is_error
        assert out.content == "(no matches)"
    finally:
        outside.unlink(missing_ok=True)
