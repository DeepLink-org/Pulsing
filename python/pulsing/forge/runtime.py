# SPDX-License-Identifier: Apache-2.0
"""In-process tool runtime — dispatches calls within a Forge environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pulsing.forge.context import ToolCallContext
from pulsing.forge.discovery.catalog import ToolCatalog
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.result import ToolResult
from pulsing.forge.session import LocalToolSession, NullToolSession, ToolSession


class LocalToolRuntime:
    """Framework-agnostic local tool dispatch."""

    def __init__(
        self,
        *,
        cwd: str = ".",
        sandbox_policy: str = "off",
        dangerously_disable_sandbox: bool = False,
        session: ToolSession | None = None,
        tool_catalog: ToolCatalog | None = None,
    ) -> None:
        catalog = tool_catalog or ToolCatalog()
        if tool_catalog is None:
            catalog.load_codex_plugins()
        self._ctx = ToolCallContext(
            cwd=Path(cwd),
            sandbox_policy=sandbox_policy,
            dangerously_disable_sandbox=dangerously_disable_sandbox,
            session=session or LocalToolSession(),
            tool_catalog=catalog,
        )

    @property
    def context(self) -> ToolCallContext:
        return self._ctx

    @property
    def session(self) -> ToolSession:
        return self._ctx.session_nonnull

    def set_code_mode(self, code_mode: Any) -> None:
        self._ctx.code_mode = code_mode

    def close(self) -> None:
        """Kill background exec sessions (PTY/subprocess) — avoid orphaned children."""
        self._ctx.exec.stop_all()

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        return dispatch_tool(name, dict(arguments or {}), ctx=self._ctx)

    async def acall_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        return self.call_tool(name, arguments)

    def tool_names(self) -> list[str]:
        from pulsing.forge.integrated import FORGE_TOOL_NAMES

        return sorted(FORGE_TOOL_NAMES)
