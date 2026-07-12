# SPDX-License-Identifier: Apache-2.0
"""Phase 2: actors and loop live in pulsing.agent."""

from __future__ import annotations

from pulsing.agent import Agent, LlmChat
from pulsing.agent.actors import AgentActor
from pulsing.agent.actors.forge_events import emit_forge_event
from pulsing.agent.loop.permissions import PermissionChecker
from pulsing.agent.loop.split_tools import build_tools_for_agent
from pulsing.agent.loop.tool_base import ToolResult


def test_workspace_agent_exports() -> None:
    from pulsing.agent.actors import Agent as WorkspaceAgent, NpcAgent
    from pulsing.agent.host import Agent as ExportedAgent

    assert ExportedAgent is WorkspaceAgent
    assert NpcAgent is WorkspaceAgent
    assert issubclass(WorkspaceAgent._cls, AgentActor)


def test_llm_chat_import() -> None:
    assert LlmChat is not None


def test_tool_result_type() -> None:
    r = ToolResult(content="ok")
    assert r.content == "ok"


def test_emit_forge_event_callable() -> None:
    assert callable(emit_forge_event)


def test_build_tools_for_agent_smoke() -> None:
    checker = PermissionChecker(auto_approve=True)
    tools = build_tools_for_agent(
        checker,
        cluster_enabled=True,
        summon_enabled=False,
        tool_allowlist={"Read", "ListClusterAgents"},
    )
    names = {t.name for t in tools}
    assert "Read" in names
    assert "ListClusterAgents" in names
