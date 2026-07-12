# SPDX-License-Identifier: Apache-2.0
"""Workspace agent resolve via ``pul.resolve`` (gossip names under ``craft/ws/``)."""

from __future__ import annotations

from typing import Any

import pulsing as pul

from pulsing.agent.cluster.constants import full_agent_name, short_agent_name
from pulsing.agent.cluster.discovery import format_agent_table, list_cluster_agents
from pulsing.agent.loop.tool_base import ToolResult


async def resolve_craft_agent(
    system: Any,
    target: str,
    *,
    workspace_id: str | None = None,
    timeout: float = 120.0,
) -> Any:
    _ = system
    if not workspace_id:
        raise ValueError("workspace_id is required")
    from pulsing.agent.actors import Agent

    short = short_agent_name(target, workspace_id=workspace_id)
    full = full_agent_name(short, workspace_id=workspace_id)
    return await pul.resolve(full, cls=Agent, timeout=timeout)


def _tool_result_from_out(out: Any, *, wait: bool) -> ToolResult:
    if isinstance(out, dict) and not out.get("ok", True):
        return ToolResult(content=str(out.get("error", out)), is_error=True)
    if isinstance(out, dict) and wait:
        body = str(out.get("assistant_text") or "")
        if not body and out.get("events"):
            body = str(out.get("events"))[:8000]
        return ToolResult(content=body or "(ok, empty reply)")
    return ToolResult(content=str(out))


async def message_cluster_agent(
    system: Any,
    *,
    target: str,
    message: str,
    from_agent: str,
    workspace_id: str | None = None,
    timeout: float = 600.0,
    wait: bool = False,
) -> ToolResult:
    """LLM tool: ``resolve`` + ``deliver_message`` (Pulsing ask RPC)."""
    text = (message or "").strip()
    if not text:
        return ToolResult(content="empty message", is_error=True)
    try:
        proxy = await resolve_agent(
            system,
            target,
            workspace_id=workspace_id,
            timeout=min(timeout, 120.0),
        )
    except Exception as e:
        return ToolResult(content=f"resolve {target!r} failed: {e!r}", is_error=True)
    try:
        out = await proxy.deliver_message(
            from_agent,
            text,
            channel="whisper",
            wait=wait,
            timeout=timeout,
        )
    except Exception as e:
        return ToolResult(content=f"deliver_message failed: {e!r}", is_error=True)
    return _tool_result_from_out(out, wait=wait)


async def dispatch_cluster_tool(
    agent: Any, name: str, kwargs: dict[str, Any]
) -> ToolResult:
    if name == "ListClusterAgents":
        local_only = bool(kwargs.get("local_only", False))
        rows = await list_cluster_agents(
            pul.get_system(),
            workspace_id=agent._workspace_id,
            local_node_only=local_only,
        )
        return ToolResult(
            content=format_agent_table(rows, workspace_id=agent._workspace_id)
        )
    if name == "MessageClusterAgent":
        target = str(
            kwargs.get("agent")
            or kwargs.get("target")
            or kwargs.get("to")
            or kwargs.get("name")
            or "",
        ).strip()
        message = str(kwargs.get("message") or kwargs.get("text") or "").strip()
        if not target:
            return ToolResult(
                content="MessageClusterAgent: agent/target/to/name is required.",
                is_error=True,
            )
        if not message:
            return ToolResult(
                content="MessageClusterAgent: message/text is required.",
                is_error=True,
            )
        wait = kwargs.get("wait", False)
        if isinstance(wait, str):
            wait = wait.strip().lower() not in ("0", "false", "no", "off")
        timeout = float(kwargs.get("timeout", 600.0))
        from_name = agent._cluster_short_name or "anonymous"
        return await message_cluster_agent(
            pul.get_system(),
            target=target,
            message=message,
            from_agent=from_name,
            workspace_id=agent._workspace_id,
            timeout=timeout,
            wait=bool(wait),
        )
    return ToolResult(content=f"unknown cluster tool: {name}", is_error=True)


# Preferred public name; ``resolve_craft_agent`` kept for backward compatibility.
resolve_agent = resolve_craft_agent

__all__ = [
    "dispatch_cluster_tool",
    "message_cluster_agent",
    "resolve_agent",
    "resolve_craft_agent",
]
