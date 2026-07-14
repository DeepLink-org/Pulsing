# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the hosted ``web_search`` Forge tool.

Codex registers web_search as a Responses API provider tool (type=web_search),
not a sandbox function. These tests pin Forge's contract: argument validation,
hosted deferral without a provider hook, and clear auth failures when Craft
wires a hook but credentials are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.extension.web_search import handlers as web_search
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.session import LocalToolSession

pytestmark = pytest.mark.forge


def _ctx(tmp_path, *, session: Any = None) -> ToolCallContext:
    return ToolCallContext(cwd=tmp_path, session=session)


@dataclass
class _HookSession(LocalToolSession):
    hook: Any = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def hosted_web_search(self, arguments: dict[str, Any]) -> Any:
        self.calls.append(dict(arguments))
        if self.hook is None:
            return None
        return self.hook(arguments)


def test_empty_query_rejected(tmp_path) -> None:
    out = dispatch_tool("web_search", {}, ctx=_ctx(tmp_path))
    assert out.is_error
    assert "query" in out.content
    assert "queries" in out.content


def test_whitespace_only_query_rejected(tmp_path) -> None:
    out = dispatch_tool("web_search", {"query": "   "}, ctx=_ctx(tmp_path))
    assert out.is_error


def test_hosted_deferred_without_hook(tmp_path) -> None:
    out = dispatch_tool("web_search", {"query": "pulsing actor"}, ctx=_ctx(tmp_path))
    assert not out.is_error
    assert out.structured is not None
    assert out.structured["kind"] == "hosted_web_search"
    assert out.structured["status"] == "deferred"
    assert out.structured["reason"] == "provider_not_configured"
    assert out.structured["arguments"] == {"query": "pulsing actor"}
    assert "Responses API" in out.content
    assert "web.run" in out.content


def test_hosted_deferred_accepts_queries_list(tmp_path) -> None:
    out = dispatch_tool(
        "web_search",
        {"queries": ["alpha", " beta "]},
        ctx=_ctx(tmp_path),
    )
    assert not out.is_error
    assert out.structured is not None
    assert out.structured["arguments"] == {"queries": ["alpha", "beta"]}


def test_hook_success_returns_structured(tmp_path) -> None:
    session = _HookSession(
        hook=lambda args: {
            "status": "ok",
            "results": [{"title": "hit", "url": "https://x"}],
        },
    )
    out = dispatch_tool(
        "web_search", {"query": "test"}, ctx=_ctx(tmp_path, session=session)
    )
    assert not out.is_error
    assert out.structured is not None
    assert out.structured["status"] == "ok"
    assert session.calls == [{"query": "test"}]


def test_hook_none_falls_back_to_deferred(tmp_path) -> None:
    session = _HookSession(hook=lambda _args: None)
    out = dispatch_tool(
        "web_search", {"query": "test"}, ctx=_ctx(tmp_path, session=session)
    )
    assert not out.is_error
    assert out.structured is not None
    assert out.structured["status"] == "deferred"


def test_hook_auth_exception_clear_message(tmp_path) -> None:
    session = _HookSession(
        hook=lambda _args: (_ for _ in ()).throw(RuntimeError("missing API key")),
    )
    out = dispatch_tool(
        "web_search", {"query": "test"}, ctx=_ctx(tmp_path, session=session)
    )
    assert out.is_error
    assert "provider auth missing" in out.content
    assert "OPENAI_API_KEY" in out.content


def test_hook_auth_error_dict_clear_message(tmp_path) -> None:
    session = _HookSession(
        hook=lambda _args: {"status": "error", "error": "401 unauthorized"},
    )
    out = dispatch_tool(
        "web_search", {"query": "test"}, ctx=_ctx(tmp_path, session=session)
    )
    assert out.is_error
    assert "provider auth missing" in out.content
    assert out.structured is not None
    assert out.structured["status"] == "error"


def test_hook_generic_exception_message(tmp_path) -> None:
    session = _HookSession(
        hook=lambda _args: (_ for _ in ()).throw(RuntimeError("upstream timeout")),
    )
    out = dispatch_tool(
        "web_search", {"query": "test"}, ctx=_ctx(tmp_path, session=session)
    )
    assert out.is_error
    assert "provider hook failed" in out.content
    assert "auth missing" not in out.content


def test_auth_hint_detects_common_patterns() -> None:
    assert web_search._AUTH_ERROR_HINT.search("missing API key")
    assert web_search._AUTH_ERROR_HINT.search("401 Unauthorized")
    assert not web_search._AUTH_ERROR_HINT.search("connection reset")
