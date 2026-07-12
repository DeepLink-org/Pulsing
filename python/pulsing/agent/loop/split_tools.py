# SPDX-License-Identifier: Apache-2.0
"""Build mixed tool list: isolated (schema-only) + parent-local tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pulsing.agent.npc.loader import list_npc_classes
from pulsing.agent.loop.cluster_tools import (
    ListClusterAgentsTool,
    MessageClusterAgentTool,
)
from pulsing.agent.loop.constants import (
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
)
from pulsing.agent.loop.forge_tools import all_forge_tool_templates
from pulsing.agent.loop.permissions import PermissionChecker
from pulsing.agent.loop.quest_tools import QuestReportTool
from pulsing.agent.loop.tool_base import Tool, ToolResult
from pulsing.agent.loop.tools_pkg import (
    FetchUrlTool,
    _json_schema_object,
)
from pulsing.core.proxy import ActorProxy


class SummonTool(Tool):
    @property
    def name(self) -> str:
        return "Summon"

    @property
    def description(self) -> str:
        return "Summon another NPC to help. Set wait=false to spawn without blocking."

    @property
    def input_schema(self) -> dict:
        return _json_schema_object(
            {
                "goal": {"type": "string"},
                "npc_class": {
                    "type": "string",
                    "description": "Npc class id (see .pulsing/npcs/*.json).",
                },
                "name": {"type": "string"},
                "personality": {"type": "string"},
                "task_id": {"type": "string"},
                "wait": {
                    "type": "boolean",
                    "description": "Block for child reply (default true).",
                },
                "timeout": {
                    "type": "number",
                    "description": "Max seconds when wait=true (default 600).",
                },
            },
            ["goal"],
        )

    def is_read_only(self) -> bool:
        return False

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("Summon runs on Agent.")


class _IsolatedSchemaTool(Tool):
    def __init__(self, template: Tool) -> None:
        self._t = template

    @property
    def name(self) -> str:
        return self._t.name

    @property
    def description(self) -> str:
        return self._t.description

    @property
    def input_schema(self) -> dict:
        return self._t.input_schema

    def is_read_only(self) -> bool:
        return self._t.is_read_only()

    def get_activity_description(self, **kwargs: Any) -> str | None:
        return self._t.get_activity_description(**kwargs)

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError(f"{self.name} runs in the isolated worker.")


def build_tools_for_agent(
    checker: PermissionChecker,
    proxy: ActorProxy | None = None,
    *,
    cwd: str = ".",
    cluster_enabled: bool = False,
    summon_enabled: bool = False,
    quest_enabled: bool = True,
    tool_allowlist: set[str] | None = None,
    tool_forbid: set[str] | None = None,
) -> list[Tool]:
    _ = checker, proxy, cwd
    out: list[Tool] = [_IsolatedSchemaTool(t) for t in all_forge_tool_templates()]
    out.append(FetchUrlTool())
    if quest_enabled:
        out.append(QuestReportTool())
    if cluster_enabled:
        out.extend([ListClusterAgentsTool(), MessageClusterAgentTool()])
    if summon_enabled:
        out.append(SummonTool())
    if tool_allowlist:
        out = [t for t in out if t.name in tool_allowlist]
    if tool_forbid:
        out = [t for t in out if t.name not in tool_forbid]
    return out


def local_tool_names(tools: list[Tool]) -> set[str]:
    return {
        t.name
        for t in tools
        if t.name not in FORGE_ISOLATED_TOOL_NAMES
        and t.name not in FORGE_HOST_TOOL_NAMES
    }


def npc_class_names(workspace_root: Path | str | None = None) -> list[str]:
    root = Path(workspace_root) if workspace_root else None
    return list_npc_classes(root)
