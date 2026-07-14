# SPDX-License-Identifier: Apache-2.0
"""Unified Forge runtime: tools + P2P session hooks on one host link."""

from __future__ import annotations

from typing import Any

from pulsing.forge.events import ForgeEvent
from pulsing.forge.p2p_session import EmitFn
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.result import ToolResult
from pulsing.forge.session import ToolSession

# Tools executed in isolated ToolWorkerActor (filesystem + execution).
FORGE_ISOLATED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "Edit",
        "Write",
        "Bash",
        "shell_command",
        "exec_command",
        "write_stdin",
        "apply_patch",
        "view_image",
    }
)

# Session tools run in-process on the host (need UI / agent state).
FORGE_HOST_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "update_plan",
        "new_context",
        "get_context_remaining",
        "request_user_input",
        "request_permissions",
        "tool_search",
        "list_available_plugins_to_install",
        "request_plugin_install",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "exec",
        "wait",
        "web.run",
        "skills.list",
        "skills.read",
        "memories.list",
        "memories.read",
        "memories.search",
        "memories.add_ad_hoc_note",
        "web_search",
    }
)

FORGE_TOOL_NAMES: frozenset[str] = FORGE_ISOLATED_TOOL_NAMES | FORGE_HOST_TOOL_NAMES


class ForgeHostLink:
    """In-process Forge runtime wired to a host agent via P2P events."""

    def __init__(
        self,
        *,
        cwd: str,
        sandbox_policy: str,
        dangerously_disable_sandbox: bool,
        session: ToolSession,
        emit: EmitFn | None = None,
    ) -> None:
        self._emit = emit
        self._runtime = LocalToolRuntime(
            cwd=cwd,
            sandbox_policy=sandbox_policy,
            dangerously_disable_sandbox=dangerously_disable_sandbox,
            session=session,
            tool_catalog=getattr(session, "tool_catalog", None),
        )

    @property
    def runtime(self) -> LocalToolRuntime:
        return self._runtime

    def close(self) -> None:
        self._runtime.close()

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        args = dict(arguments or {})
        if self._emit is not None:
            self._emit(ForgeEvent.tool_begin(name, args))
        out = self._runtime.call_tool(name, args)
        if self._emit is not None:
            self._emit(
                ForgeEvent.tool_end(
                    name,
                    is_error=bool(out.is_error),
                    content_preview=out.content,
                )
            )
        return out
