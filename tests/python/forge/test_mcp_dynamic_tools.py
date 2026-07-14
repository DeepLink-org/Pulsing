# SPDX-License-Identifier: Apache-2.0
"""MCP dynamic function tools — routing, wire schema, hybrid integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.hybrid_runtime import HybridForgeRuntime
from pulsing.forge.mcp.manager import McpManager
from pulsing.forge.mcp.naming import MCP_TOOL_NAME_PREFIX, is_mcp_dynamic_tool
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE

pytestmark = pytest.mark.forge


def test_is_mcp_dynamic_tool() -> None:
    assert is_mcp_dynamic_tool("mcp__github__search")
    assert not is_mcp_dynamic_tool("list_mcp_resources")
    assert not is_mcp_dynamic_tool("Read")


def test_mcp_manager_sync_live_tools_from_rust() -> None:
    class _Rust:
        def mcp_tool_specs(self) -> list[dict]:
            return [
                {
                    "name": "mcp__demo__echo",
                    "description": "Echo",
                    "input_schema": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                    },
                    "server_name": "demo",
                    "tool_name": "echo",
                }
            ]

    mgr = McpManager()
    mgr.sync_live_tools_from_rust(_Rust())
    stubs = mgr.deferred_tool_stubs()
    assert len(stubs) == 1
    assert stubs[0].model_name == "mcp__demo__echo"
    assert stubs[0].input_schema["properties"]["msg"]["type"] == "string"


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_hybrid_mcp_dynamic_tool_routes_to_rust(tmp_path: Path) -> None:
    rt = HybridForgeRuntime.create(cwd=str(tmp_path), auto_approve=True)
    name = f"{MCP_TOOL_NAME_PREFIX}missing__tool"
    out = rt.call_tool(name, {})
    assert not out.content.startswith("Unknown tool:")
    assert "unknown MCP tool" in out.content.lower() or "MCP tool" in out.content


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_hybrid_mcp_tool_specs_wire_schema(tmp_path: Path) -> None:
    rt = HybridForgeRuntime.create(cwd=str(tmp_path), auto_approve=True)
    rust = rt.rust_runtime
    assert rust is not None
    specs = rust.mcp_tool_specs()
    assert isinstance(specs, list)
    for spec in specs:
        assert spec["name"].startswith(MCP_TOOL_NAME_PREFIX)
        schema = spec.get("input_schema") or {}
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)
