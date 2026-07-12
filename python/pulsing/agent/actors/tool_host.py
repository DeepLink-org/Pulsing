# SPDX-License-Identifier: Apache-2.0
"""Tool dispatch + isolated worker."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pulsing.agent.actors.activity import set_activity
from pulsing.agent.actors.forge_events import emit_forge_event
from pulsing.agent.cluster.resolve import dispatch_cluster_tool
from pulsing.forge.backend import ForgeBackend, ForgeIsolatedWorker
from pulsing.forge.config import ToolWorkerConfig
from pulsing.forge.discovery.activate import activate_discovered_tools
from pulsing.forge.events import ForgeEvent
from pulsing.agent.loop.constants import (
    CLUSTER_TOOL_NAMES,
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
    NPC_TOOL_NAMES,
    QUEST_TOOL_NAMES,
)
from pulsing.agent.loop.tool_base import ToolResult

logger = logging.getLogger(__name__)


def _worker_config(agent: Any) -> ToolWorkerConfig:
    return ToolWorkerConfig(
        cwd=agent._cwd,
        sandbox_policy=agent._sandbox_policy,
        dangerously_disable_sandbox=agent._dangerously_disable_sandbox,
        auto_approve=bool(getattr(agent, "_auto_approve", False)),
        event_sink_name=getattr(agent, "_event_sink_name", None),
        host_name=getattr(agent, "_forge_host_name", None),
    )


def _tool_activity(agent: Any, name: str, kwargs: dict[str, Any]) -> str:
    tool = agent._tools_by_name.get(name)
    if tool is None:
        return name
    return tool.get_activity_description(**kwargs) or name


def _forge_backend(agent: Any) -> ForgeBackend:
    backend = getattr(agent, "_forge_backend", None)
    if backend is not None:
        return backend
    host = getattr(agent, "_forge_host", None)
    if host is None:
        raise RuntimeError("Forge host runtime not initialized")
    worker = getattr(agent, "_forge_worker", None)
    backend = ForgeBackend(
        host=host,
        worker=worker,
        event_sink_name=getattr(agent, "_event_sink_name", None),
    )
    agent._forge_backend = backend
    return backend


async def _ensure_forge_worker(
    agent: Any, *, reason: str = "startup"
) -> ForgeIsolatedWorker:
    worker = getattr(agent, "_forge_worker", None)
    if worker is None:
        cfg = _worker_config(agent)
        if agent._shared_tool_worker and agent._workspace_id:
            worker = ForgeIsolatedWorker.shared(cfg, workspace_id=agent._workspace_id)
        else:
            worker = ForgeIsolatedWorker.dedicated(cfg)
        agent._forge_worker = worker
        agent._forge_backend = None
    await worker.ensure_ready(reason=reason)
    return worker


async def call_tool(agent: Any, name: str, kwargs: dict[str, Any]) -> ToolResult:
    set_activity(
        agent, state="tool", detail=_tool_activity(agent, name, kwargs), tool=name
    )
    try:
        if name in CLUSTER_TOOL_NAMES:
            return await dispatch_cluster_tool(agent, name, kwargs)
        if name in NPC_TOOL_NAMES:
            from pulsing.agent.actors.summon_tool import tool_summon

            return await tool_summon(agent, kwargs)
        if name in QUEST_TOOL_NAMES:
            from pulsing.agent.loop.quest_tools import tool_quest_report

            return await tool_quest_report(agent, kwargs)
        if name in FORGE_HOST_TOOL_NAMES:
            return await _host_forge_tool(agent, name, kwargs)
        if name in FORGE_ISOLATED_TOOL_NAMES:
            return await _isolated_forge_tool(agent, name, kwargs)
        tool = agent._local_tools.get(name)
        if tool is None:
            return ToolResult(content=f"Unknown tool: {name}", is_error=True)
        await emit_forge_event(agent, ForgeEvent.tool_begin(name, kwargs))
        out = ToolResult(content="", is_error=True)
        try:
            out = await asyncio.to_thread(tool.execute, **kwargs)
        finally:
            await emit_forge_event(
                agent,
                ForgeEvent.tool_end(
                    name,
                    is_error=bool(out.is_error),
                    content_preview=out.content,
                ),
            )
        return out
    finally:
        set_activity(agent, state="thinking", detail="LLM turn", tool="")


async def ensure_isolated_worker(agent: Any, *, reason: str = "startup") -> None:
    await _ensure_forge_worker(agent, reason=reason)


async def _host_forge_tool(agent: Any, name: str, kwargs: dict[str, Any]) -> ToolResult:
    backend = _forge_backend(agent)
    result = await backend.call_tool(name, kwargs)
    if not result.is_error:
        if name == "tool_search":
            llm = getattr(agent, "_llm", None)
            activate_discovered_tools(
                agent._tools_by_name,
                result.content,
                register=llm.register_tool if llm is not None else None,
            )
        elif name == "request_plugin_install":
            _sync_installed_plugin_tools(agent)
            await _refresh_mcp_runtime(agent)
    return ToolResult(content=result.content, is_error=result.is_error)


async def _isolated_forge_tool(
    agent: Any, name: str, kwargs: dict[str, Any]
) -> ToolResult:
    await _ensure_forge_worker(agent, reason="tool call")
    backend = _forge_backend(agent)
    return await backend.call_tool(name, kwargs)


def _sync_installed_plugin_tools(agent: Any) -> None:
    session = getattr(agent, "_forge_session", None)
    if session is None:
        return
    llm = getattr(agent, "_llm", None)
    for entry in session.tool_catalog.deferred:
        if entry.name in agent._tools_by_name:
            continue
        from pulsing.forge.discovery.activate import DeferredDiscoveredTool

        tool = DeferredDiscoveredTool(
            name=entry.name,
            description=entry.description,
            parameters=entry.parameters,
        )
        agent._tools_by_name[entry.name] = tool
        if llm is not None:
            llm.register_tool(tool)


def _refresh_mcp_runtime(agent: Any) -> None:
    backend = getattr(agent, "_forge_backend", None)
    if backend is not None:
        backend.refresh_mcp()
    session = getattr(agent, "_forge_session", None)
    if session is not None:
        try:
            session.tool_catalog.refresh_from_codex()
        except Exception as exc:
            logger.warning("MCP catalog refresh failed: %s", exc)


async def refresh_mcp_tools(agent: Any) -> list[str]:
    from pulsing.forge.mcp.sync import sync_mcp_tools_to_agent

    _refresh_mcp_runtime(agent)
    return await sync_mcp_tools_to_agent(agent)


async def stop_isolated_worker(agent: Any) -> None:
    """Tear down the isolated worker + host runtime (avoid orphaned exec children)."""
    worker = getattr(agent, "_forge_worker", None)
    if worker is not None:
        try:
            await worker.close()
        except Exception as exc:
            logger.warning("isolated worker close failed: %s", exc)
    host = getattr(agent, "_forge_host", None)
    if host is not None and hasattr(host, "close"):
        try:
            host.close()
        except Exception as exc:
            logger.warning("forge host close failed: %s", exc)
