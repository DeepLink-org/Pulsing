# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible wrapper — prefer tests/python/forge/test_gates.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.hybrid_runtime import HybridForgeRuntime
from pulsing.forge.integrated import FORGE_TOOL_NAMES
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE
from pulsing.testing.forge_harness import minimal_tool_args

pytestmark = pytest.mark.forge


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
@pytest.mark.forge_l1
def test_hybrid_all_tools_callable(tmp_path: Path) -> None:
    rt = HybridForgeRuntime.create(cwd=str(tmp_path), auto_approve=True)
    names = sorted(FORGE_TOOL_NAMES)
    assert len(names) == 32
    failures: list[str] = []
    for name in names:
        out = rt.call_tool(name, minimal_tool_args(name, tmp_path))
        if out.content.startswith("Unknown tool:"):
            failures.append(name)
    assert not failures, f"Unknown tool for: {failures}"


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
@pytest.mark.forge_l1
def test_hybrid_mcp_list_resources(tmp_path: Path) -> None:
    rt = HybridForgeRuntime.create(cwd=str(tmp_path), auto_approve=True)
    out = rt.call_tool("list_mcp_resources", {})
    assert not out.content.startswith("Unknown tool:")
    assert "MCP runtime is not initialized" not in out.content
    assert "MCP runtime is not started" not in out.content


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
@pytest.mark.forge_l1
def test_hybrid_list_mcp_resource_templates_validates_args(tmp_path: Path) -> None:
    rt = HybridForgeRuntime.create(cwd=str(tmp_path), auto_approve=True)
    missing_server = rt.call_tool("list_mcp_resource_templates", {})
    assert missing_server.is_error
    assert "server is required" in missing_server.content

    empty_server = rt.call_tool("list_mcp_resource_templates", {"server": "  "})
    assert empty_server.is_error
    assert "server must be a non-empty string" in empty_server.content

    wired = rt.call_tool(
        "list_mcp_resource_templates",
        {"server": "demo"},
    )
    assert not wired.content.startswith("Unknown tool:")
    assert "MCP runtime is not initialized" not in wired.content


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
@pytest.mark.forge_l1
def test_hybrid_read_mcp_resource_validates_args(tmp_path: Path) -> None:
    rt = HybridForgeRuntime.create(cwd=str(tmp_path), auto_approve=True)
    missing_server = rt.call_tool("read_mcp_resource", {"uri": "file:///dev/null"})
    assert missing_server.is_error
    assert "server is required" in missing_server.content

    bad_uri = rt.call_tool(
        "read_mcp_resource",
        {"server": "demo", "uri": "not-a-uri"},
    )
    assert bad_uri.is_error
    assert "invalid uri" in bad_uri.content.lower()

    wired = rt.call_tool(
        "read_mcp_resource",
        {"server": "demo", "uri": "file:///dev/null"},
    )
    assert not wired.content.startswith("Unknown tool:")
    assert "MCP runtime is not initialized" not in wired.content
