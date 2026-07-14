# SPDX-License-Identifier: Apache-2.0
"""MCP dynamic tool naming — aligned with ``pulsing-forge`` ``LEGACY_MCP_TOOL_NAME_PREFIX``."""

from __future__ import annotations

MCP_TOOL_NAME_PREFIX = "mcp__"


def is_mcp_dynamic_tool(name: str) -> bool:
    """True when ``name`` is a per-server MCP function tool (not a Forge builtin)."""
    return name.startswith(MCP_TOOL_NAME_PREFIX)
