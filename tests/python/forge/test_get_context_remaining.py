# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the ``get_context_remaining`` Forge tool.

Handler contract (Rust + Python): ``{"tokens_remaining": <int|null>, "status": "ok"|"unknown"}``.
Token counts come from ``ToolSession.tokens_remaining()`` — static ``token_budget`` on
``LocalToolSession``, or host-side estimate via ``AgentForgeSession`` / hybrid callback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE
from pulsing.forge.session import LocalToolSession, NullToolSession
from pulsing.testing.forge_harness import local_runtime

pytestmark = [pytest.mark.forge, pytest.mark.forge_l2]


def _payload(out) -> dict[str, Any]:
    return json.loads(out.content)


def test_known_budget_returns_ok(tmp_path: Path) -> None:
    ctx = ToolCallContext(cwd=tmp_path, session=LocalToolSession(token_budget=42_000))
    out = dispatch_tool("get_context_remaining", {}, ctx=ctx)
    assert not out.is_error
    assert _payload(out) == {"tokens_remaining": 42_000, "status": "ok"}


def test_null_session_degrades_to_unknown(tmp_path: Path) -> None:
    ctx = ToolCallContext(cwd=tmp_path, session=None)
    assert isinstance(ctx.session, NullToolSession)
    out = dispatch_tool("get_context_remaining", {}, ctx=ctx)
    assert not out.is_error
    assert _payload(out) == {"tokens_remaining": None, "status": "unknown"}


def test_local_session_without_budget_returns_unknown(tmp_path: Path) -> None:
    ctx = ToolCallContext(cwd=tmp_path, session=LocalToolSession())
    out = dispatch_tool("get_context_remaining", {}, ctx=ctx)
    assert not out.is_error
    assert _payload(out) == {"tokens_remaining": None, "status": "unknown"}


def test_local_forge_default_session_reports_budget(tmp_path: Path) -> None:
    rt = local_runtime(tmp_path)
    out = rt.call_tool("get_context_remaining", {})
    assert not out.is_error
    payload = _payload(out)
    assert payload["status"] == "ok"
    assert payload["tokens_remaining"] == 128_000


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_hybrid_reads_session_token_budget(tmp_path: Path) -> None:
    from pulsing.forge.hybrid_runtime import HybridForgeRuntime

    rt = HybridForgeRuntime.create(
        cwd=str(tmp_path),
        auto_approve=True,
        session=LocalToolSession(token_budget=99_000),
    )
    out = rt.call_tool("get_context_remaining", {})
    assert not out.is_error
    payload = _payload(out)
    assert payload["status"] == "ok"
    assert payload["tokens_remaining"] == 99_000


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_hybrid_python_payload_semantics_match(tmp_path: Path) -> None:
    from pulsing.forge.hybrid_runtime import HybridForgeRuntime

    session = LocalToolSession(token_budget=55_000)
    ctx = ToolCallContext(cwd=tmp_path, session=session)
    py_out = dispatch_tool("get_context_remaining", {}, ctx=ctx)
    hybrid = HybridForgeRuntime.create(
        cwd=str(tmp_path),
        auto_approve=True,
        session=session,
    )
    rust_out = hybrid.call_tool("get_context_remaining", {})
    assert _payload(py_out) == _payload(rust_out)


def test_agent_session_estimate_without_static_budget() -> None:
    from pulsing.agent.actors.forge_session import AgentForgeSession

    llm = MagicMock()
    llm.estimate_tokens_remaining.return_value = 12_345
    agent = MagicMock(_llm=llm)
    session = AgentForgeSession(context_window=100_000)
    session._agent = agent  # noqa: SLF001 — host back-ref
    assert session.tokens_remaining() == 12_345
    llm.estimate_tokens_remaining.assert_called_once_with(100_000)


def test_llm_estimate_tokens_remaining_from_transcript() -> None:
    from pulsing.agent.loop.llm_chat import LlmChat

    llm = LlmChat(
        backend=MagicMock(),
        tools={},
        permission_checker=MagicMock(),
        model="test",
        max_tokens=1000,
    )
    llm._messages = [{"role": "user", "content": "x" * 400}]
    remaining = llm.estimate_tokens_remaining(context_window=10_000)
    assert remaining == 10_000 - 100 - 1000
