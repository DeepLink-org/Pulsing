# SPDX-License-Identifier: Apache-2.0
"""Workspace MCP hub actor — refresh connections and list live tools."""

from __future__ import annotations

import logging
from typing import Any

from pulsing.core.proxy import ActorProxy
from pulsing.core.remote import remote, resolve

from pulsing.forge.mcp.manager import get_global_mcp_manager
from pulsing.forge.naming import mcp_hub_name

logger = logging.getLogger(__name__)


@remote
class McpHubActor:
    """Owns a Forge host runtime for MCP refresh and tool discovery."""

    def __init__(self, cwd: str = ".", *, auto_approve: bool = True) -> None:
        from pulsing.forge.backend import ForgeHostConfig, create_host_runtime

        self._cwd = cwd
        self._host = create_host_runtime(
            ForgeHostConfig(cwd=cwd, auto_approve=auto_approve),
        )

    async def refresh(self) -> dict[str, Any]:
        self._host.refresh_mcp()
        tools = self.list_tools()
        return {"ok": True, "tools": tools, "count": len(tools)}

    def list_tools(self) -> list[dict[str, Any]]:
        rust = getattr(self._host, "rust_runtime", None)
        mgr = get_global_mcp_manager()
        mgr.refresh_catalog()
        if rust is not None:
            mgr.sync_live_tools_from_rust(rust)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for stub in mgr.deferred_tool_stubs():
            if stub.model_name in seen:
                continue
            seen.add(stub.model_name)
            out.append(
                {
                    "name": stub.model_name,
                    "description": stub.description or stub.model_name,
                    "parameters": dict(
                        stub.input_schema or {"type": "object", "properties": {}}
                    ),
                }
            )
        return out


async def ensure_mcp_hub(workspace_id: str, *, cwd: str = ".") -> ActorProxy:
    name = mcp_hub_name(workspace_id)
    try:
        return await resolve(name, cls=McpHubActor, timeout=30.0)
    except Exception:
        return await McpHubActor.spawn(cwd, name=name, public=True)
