# SPDX-License-Identifier: Apache-2.0
"""Gossip names for Forge actors on the Pulsing cluster."""

from __future__ import annotations

FORGE_WORKER_SHORT = "_tools"
MCP_HUB_SHORT = "_mcp_hub"
# Gossip namespace for workspace-scoped agents and shared workers.
# Legacy ``craft/ws`` remains the default until a coordinated prefix migration.
DEFAULT_WORKSPACE_PREFIX = "craft/ws"
FUTURE_WORKSPACE_PREFIX = "agent/ws"


def shared_tool_worker_name(
    workspace_id: str,
    *,
    prefix: str = DEFAULT_WORKSPACE_PREFIX,
) -> str:
    """Public gossip name for a workspace-level shared ``ToolWorkerActor``."""
    ws = (workspace_id or "").strip().strip("/")
    if not ws:
        raise ValueError("workspace_id must be non-empty")
    p = (prefix or DEFAULT_WORKSPACE_PREFIX).strip().strip("/")
    return f"{p}/{ws}/{FORGE_WORKER_SHORT}"


def forge_event_inbox_name(host_name: str) -> str:
    """Named inbox actor that receives Forge tell events for a host agent."""
    host = (host_name or "").strip().strip("/")
    if not host:
        raise ValueError("host_name must be non-empty")
    return f"{host}/events"


def worker_supervisor_name(host_name: str) -> str:
    """In-process supervisor that wraps an isolated ``ToolWorkerActor``."""
    host = (host_name or "").strip().strip("/")
    if not host:
        raise ValueError("host_name must be non-empty")
    return f"{host}/worker"


def mcp_hub_name(
    workspace_id: str,
    *,
    prefix: str = DEFAULT_WORKSPACE_PREFIX,
) -> str:
    """Workspace-level MCP connection hub."""
    ws = (workspace_id or "").strip().strip("/")
    if not ws:
        raise ValueError("workspace_id must be non-empty")
    p = (prefix or DEFAULT_WORKSPACE_PREFIX).strip().strip("/")
    return f"{p}/{ws}/{MCP_HUB_SHORT}"


def code_cell_registry_name(host_name: str) -> str:
    """Per-host Code Mode cell registry."""
    host = (host_name or "").strip().strip("/")
    if not host:
        raise ValueError("host_name must be non-empty")
    return f"{host}/code_cells"
