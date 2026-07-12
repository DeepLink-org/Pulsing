# SPDX-License-Identifier: Apache-2.0
"""Hybrid MCP resource listing — list_mcp_resources behavior."""

from __future__ import annotations

import json

import pytest

from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE

pytestmark = [
    pytest.mark.forge,
    pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop"),
]


@pytest.mark.forge_l1
def test_list_mcp_resources_returns_json_object(hybrid_forge) -> None:
    out = hybrid_forge.call_tool("list_mcp_resources", {})
    assert not out.is_error
    assert "MCP runtime is not initialized" not in out.content
    assert "MCP runtime is not started" not in out.content
    parsed = json.loads(out.content)
    assert isinstance(parsed, dict)


@pytest.mark.forge_l1
def test_list_mcp_resources_cursor_requires_server(hybrid_forge) -> None:
    out = hybrid_forge.call_tool("list_mcp_resources", {"cursor": "next"})
    assert out.is_error
    assert "server" in out.content.lower()


@pytest.mark.forge_l1
def test_list_mcp_resources_unknown_server(hybrid_forge) -> None:
    out = hybrid_forge.call_tool("list_mcp_resources", {"server": "missing-forge-test"})
    assert out.is_error
    assert "not connected" in out.content.lower()


@pytest.mark.forge_l1
def test_list_mcp_resources_rejects_blank_server(hybrid_forge) -> None:
    out = hybrid_forge.call_tool("list_mcp_resources", {"server": "   "})
    assert out.is_error
    assert "non-empty" in out.content.lower()


@pytest.mark.forge_l1
def test_refresh_mcp_keeps_list_resources_callable(hybrid_forge) -> None:
    hybrid_forge.refresh_mcp()
    out = hybrid_forge.call_tool("list_mcp_resources", {})
    assert not out.is_error
    assert json.loads(out.content) == {}
