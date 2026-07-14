# SPDX-License-Identifier: Apache-2.0
"""Forge tool registry coverage checks (CI / tests)."""

from __future__ import annotations

from pulsing.forge.handlers import _ALL as _HANDLER_TOOL_NAMES
from pulsing.forge.integrated import (
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
    FORGE_TOOL_NAMES,
)

# Host tools routed via Rust MCP / approval RPC rather than Python handlers.
_RPC_HOST_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "request_permissions",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
    },
)


def forge_dispatch_tool_names() -> frozenset[str]:
    return _HANDLER_TOOL_NAMES | _RPC_HOST_TOOL_NAMES


def assert_forge_tool_coverage() -> None:
    """Every registered Forge tool must partition cleanly and be dispatchable."""
    assert FORGE_TOOL_NAMES == FORGE_ISOLATED_TOOL_NAMES | FORGE_HOST_TOOL_NAMES
    assert not (FORGE_ISOLATED_TOOL_NAMES & FORGE_HOST_TOOL_NAMES)

    dispatchable = forge_dispatch_tool_names()
    assert dispatchable == FORGE_TOOL_NAMES, (
        sorted(dispatchable - FORGE_TOOL_NAMES),
        sorted(FORGE_TOOL_NAMES - dispatchable),
    )
