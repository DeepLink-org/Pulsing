# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the ``new_context`` tool (Python fallback path)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.handlers import NEW_CONTEXT_MESSAGE, dispatch_tool
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.session import LocalToolSession

pytestmark = [pytest.mark.forge, pytest.mark.forge_l2]

_RUST_SESSION_RS = (
    Path(__file__).resolve().parents[3] / "crates/pulsing-forge/src/handlers/session.rs"
)


def test_new_context_message_matches_rust_constant() -> None:
    text = _RUST_SESSION_RS.read_text(encoding="utf-8")
    match = re.search(r'pub const NEW_CONTEXT_MESSAGE: &str =\s*\n?\s*"([^"]+)";', text)
    assert match is not None, "NEW_CONTEXT_MESSAGE not found in session.rs"
    assert match.group(1) == NEW_CONTEXT_MESSAGE


def test_new_context_returns_message_and_flags_session(tmp_path: Path) -> None:
    session = LocalToolSession()
    rt = LocalToolRuntime(cwd=str(tmp_path), session=session)

    out = rt.call_tool("new_context", {})

    assert not out.is_error
    assert out.content == NEW_CONTEXT_MESSAGE
    assert session.new_context_requested is True


def test_new_context_without_session_does_not_raise(tmp_path: Path) -> None:
    """Missing session falls back to NullToolSession — call must still succeed."""
    ctx = ToolCallContext(cwd=tmp_path, session=None)

    out = dispatch_tool("new_context", {}, ctx=ctx)

    assert not out.is_error
    assert out.content == NEW_CONTEXT_MESSAGE


def test_new_context_propagates_session_error(tmp_path: Path) -> None:
    session = LocalToolSession()

    def fail() -> None:
        raise RuntimeError("session unavailable")

    session.request_new_context = fail  # type: ignore[method-assign]
    ctx = ToolCallContext(cwd=tmp_path, session=session)

    out = dispatch_tool("new_context", {}, ctx=ctx)

    assert out.is_error
    assert out.content == "session unavailable"
