# SPDX-License-Identifier: Apache-2.0
"""MCP connection manager — Python host side (Craft session lifecycle)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pulsing.forge.mcp.catalog import McpCatalogSnapshot, load_mcp_catalog
from pulsing.forge.mcp.naming import MCP_TOOL_NAME_PREFIX

logger = logging.getLogger(__name__)

_GLOBAL: McpManager | None = None


@dataclass
class McpToolStub:
    model_name: str
    server_name: str
    tool_name: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class McpManager:
    """Host-side MCP state. Live stdio/HTTP connections run in Rust `pulsing-forge` MCP module."""

    catalog: McpCatalogSnapshot = field(default_factory=McpCatalogSnapshot)
    live_tools: list[McpToolStub] = field(default_factory=list)
    startup_failures: list[dict[str, str]] = field(default_factory=list)
    started: bool = False

    def refresh_catalog(self) -> None:
        self.catalog = load_mcp_catalog()

    def deferred_tool_stubs(self) -> list[McpToolStub]:
        """Live MCP tools when synced from Rust; else per-server namespace stubs."""
        if self.live_tools:
            return list(self.live_tools)
        out: list[McpToolStub] = []
        for name, entry in sorted(self.catalog.servers.items()):
            if entry.config.get("enabled") is False:
                continue
            ns = f"{MCP_TOOL_NAME_PREFIX}{name}"
            out.append(
                McpToolStub(
                    model_name=ns,
                    server_name=name,
                    tool_name=ns,
                    description=f"MCP server {name} ({entry.source})",
                    input_schema={"type": "object", "properties": {}},
                )
            )
        return out

    def sync_live_tools_from_rust(self, rust: Any) -> None:
        """Pull model-visible MCP function tools from ``RustForgeAdapter``."""
        specs = []
        if hasattr(rust, "mcp_tool_specs"):
            specs = list(rust.mcp_tool_specs())
        self.live_tools = [
            McpToolStub(
                model_name=str(spec.get("name") or ""),
                server_name=str(spec.get("server_name") or ""),
                tool_name=str(spec.get("tool_name") or ""),
                description=str(spec.get("description") or spec.get("name") or ""),
                input_schema=dict(
                    spec.get("input_schema") or {"type": "object", "properties": {}}
                ),
            )
            for spec in specs
            if spec.get("name")
        ]

    async def start(self) -> None:
        """Start MCP servers (Rust runtime when maturin binding is wired)."""
        self.refresh_catalog()
        self.live_tools = self.deferred_tool_stubs()
        self.started = True
        logger.info(
            "MCP catalog loaded: %d servers (live handshake via pulsing-forge mcp)",
            len(self.catalog.servers),
        )

    async def stop(self) -> None:
        self.started = False
        self.live_tools = []


def get_global_mcp_manager() -> McpManager:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = McpManager()
    return _GLOBAL


async def refresh_global_mcp_manager() -> McpManager:
    mgr = get_global_mcp_manager()
    await mgr.start()
    return mgr
