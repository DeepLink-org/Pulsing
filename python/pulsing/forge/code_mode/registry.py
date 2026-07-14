# SPDX-License-Identifier: Apache-2.0
"""Code Mode cell registry actor."""

from __future__ import annotations

from typing import Any

from pulsing.core.proxy import ActorProxy
from pulsing.core.remote import remote, resolve

from pulsing._async_bridge import run_sync
from pulsing.forge.code_mode.protocol import WaitArgs
from pulsing.forge.code_mode.service import CodeModeService
from pulsing.forge.code_mode.tools_bridge import ToolsBridge
from pulsing.forge.naming import code_cell_registry_name
from pulsing.forge.result import ToolResult


@remote
class CodeCellRegistryActor:
    """Session-scoped code cells with nested tool calls routed to the host agent."""

    def __init__(self, host_name: str) -> None:
        self._host_name = (host_name or "").strip()
        self._service = CodeModeService()

    async def execute(
        self, source: str, *, host_name: str | None = None
    ) -> dict[str, Any]:
        host = (host_name or self._host_name).strip()
        bridge = ToolsBridge(self._host_call_tool_sync(host))
        response = self._service.execute(source, bridge)
        return response.to_dict()

    async def wait(self, args: dict[str, Any]) -> dict[str, Any]:
        wait_args = WaitArgs.from_dict(dict(args))
        response = self._service.wait(wait_args)
        return response.to_dict()

    def _host_call_tool_sync(self, host_name: str):
        def _call(name: str, args: dict[str, Any]) -> ToolResult:
            raw = run_sync(
                self._host_call_tool_async(host_name, name, args), timeout=600.0
            )
            if isinstance(raw, ToolResult):
                return raw
            return ToolResult.from_dict(dict(raw))

        return _call

    async def _host_call_tool_async(
        self,
        host_name: str,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        proxy = await resolve(host_name, timeout=60.0)
        out = await proxy.call_tool(name, dict(args))
        if isinstance(out, ToolResult):
            return out.to_dict()
        if hasattr(out, "content") and hasattr(out, "is_error"):
            return {"content": str(out.content), "is_error": bool(out.is_error)}
        if isinstance(out, dict):
            return dict(out)
        return ToolResult(content=str(out), is_error=False).to_dict()


async def ensure_code_cell_registry(host_name: str) -> ActorProxy:
    name = code_cell_registry_name(host_name)
    try:
        return await resolve(name, cls=CodeCellRegistryActor, timeout=30.0)
    except Exception:
        return await CodeCellRegistryActor.spawn(host_name, name=name, public=False)
