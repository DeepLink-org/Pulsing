# SPDX-License-Identifier: Apache-2.0
"""Register MCP tools discovered by the hub into Craft LLM schema."""

from __future__ import annotations

import logging
from typing import Any

import pulsing as pul

from pulsing.forge.discovery.deferred import DeferredForgeTool

logger = logging.getLogger(__name__)


async def sync_mcp_tools_to_agent(agent: Any) -> list[str]:
    """Refresh hub (if configured) and register MCP tools on the agent LLM."""
    hub_name = getattr(agent, "_mcp_hub_name", None)
    specs: list[dict[str, Any]] = []
    if hub_name:
        try:
            hub = await pul.resolve(hub_name, timeout=30.0)
            out = await hub.refresh()
            if isinstance(out, dict):
                specs = list(out.get("tools") or [])
        except Exception as exc:
            logger.warning("MCP hub refresh failed: %s", exc)
    if not specs:
        backend = getattr(agent, "_forge_backend", None)
        if backend is not None:
            backend.refresh_mcp()
            host = getattr(backend, "host", None)
            rust = getattr(host, "rust_runtime", None) if host is not None else None
            if rust is not None:
                from pulsing.forge.mcp.manager import get_global_mcp_manager

                get_global_mcp_manager().sync_live_tools_from_rust(rust)
        from pulsing.forge.mcp.manager import get_global_mcp_manager

        mgr = get_global_mcp_manager()
        mgr.refresh_catalog()
        specs = [
            {
                "name": stub.model_name,
                "description": stub.description or stub.model_name,
                "parameters": dict(
                    stub.input_schema or {"type": "object", "properties": {}}
                ),
            }
            for stub in mgr.deferred_tool_stubs()
        ]

    llm = getattr(agent, "_llm", None)
    activated: list[str] = []
    for spec in specs:
        name = str(spec.get("name") or "").strip()
        if not name or name in agent._tools_by_name:
            continue
        tool = DeferredForgeTool(
            name=name,
            description=str(spec.get("description") or name),
            parameters=dict(
                spec.get("parameters") or {"type": "object", "properties": {}}
            ),
        )
        agent._tools_by_name[name] = tool
        if llm is not None:
            llm.register_tool(tool)
        activated.append(name)
    return activated
