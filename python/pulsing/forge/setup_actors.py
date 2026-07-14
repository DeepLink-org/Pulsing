# SPDX-License-Identifier: Apache-2.0
"""Bootstrap Forge Pulsing actors for a Craft host agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pulsing.forge.code_mode.remote import RemoteCodeModeClient
from pulsing.forge.code_mode.registry import ensure_code_cell_registry
from pulsing.forge.event_inbox import ensure_forge_event_inbox
from pulsing.forge.mcp.hub import ensure_mcp_hub
from pulsing.forge.naming import (
    code_cell_registry_name,
    forge_event_inbox_name,
    mcp_hub_name,
)

logger = logging.getLogger(__name__)


async def ensure_forge_actors(agent: Any) -> None:
    """Spawn inbox / MCP hub / code registry and wire host runtime helpers."""
    if getattr(agent, "_forge_actors_ready", False):
        return

    host = getattr(agent, "_forge_host_name", None)
    if not host:
        agent._forge_actors_ready = True
        return

    # Independent actor lookups (each with its own resolve timeout) — run
    # concurrently so startup latency isn't the sum of all three timeouts.
    await asyncio.gather(
        _setup_inbox(agent, host),
        _setup_mcp_hub(agent),
        _setup_code_registry(agent, host),
    )

    try:
        from pulsing.forge.mcp.sync import sync_mcp_tools_to_agent

        await sync_mcp_tools_to_agent(agent)
    except Exception as exc:
        logger.debug("MCP tool sync skipped: %s", exc)

    worker = getattr(agent, "_forge_worker", None)
    if worker is not None:
        worker._cfg.event_sink_name = agent._event_sink_name
        worker._cfg.host_name = host
    agent._forge_backend = None
    agent._forge_actors_ready = True
    logger.debug("forge actors ready host=%s inbox=%s", host, agent._event_sink_name)


async def _setup_inbox(agent: Any, host: str) -> None:
    try:
        inbox = await ensure_forge_event_inbox(host)
        agent._forge_inbox_proxy = inbox
        agent._event_sink_name = forge_event_inbox_name(host)
    except Exception as exc:
        logger.warning("forge inbox setup failed, falling back to host tell: %s", exc)
        agent._event_sink_name = host


async def _setup_mcp_hub(agent: Any) -> None:
    ws = getattr(agent, "_workspace_id", None)
    if not ws:
        return
    try:
        agent._mcp_hub_name = mcp_hub_name(ws)
        await ensure_mcp_hub(ws, cwd=getattr(agent, "_cwd", "."))
    except Exception as exc:
        logger.warning("MCP hub setup failed: %s", exc)
        agent._mcp_hub_name = None


async def _setup_code_registry(agent: Any, host: str) -> None:
    try:
        registry_name = code_cell_registry_name(host)
        agent._code_cell_registry_name = registry_name
        await ensure_code_cell_registry(host)
        _wire_remote_code_mode(agent, host_name=host, registry_name=registry_name)
    except Exception as exc:
        logger.warning("code cell registry setup failed: %s", exc)
        agent._code_cell_registry_name = None


def _wire_remote_code_mode(agent: Any, *, host_name: str, registry_name: str) -> None:
    host_rt = getattr(agent, "_forge_host", None)
    if host_rt is None:
        return
    client = RemoteCodeModeClient(registry_name, host_name)
    py_rt = getattr(host_rt, "python_runtime", None)
    if py_rt is not None and hasattr(py_rt, "set_code_mode"):
        py_rt.set_code_mode(client)
