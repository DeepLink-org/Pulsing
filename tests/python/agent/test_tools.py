# SPDX-License-Identifier: Apache-2.0
"""Craft tools, isolated workers, tool_result helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_tool_worker_actor_read(tmp_path: Path) -> None:
    from pulsing.forge import ToolWorkerActor, ToolWorkerConfig

    p = tmp_path / "x.txt"
    p.write_text("hello", encoding="utf-8")
    out = ToolWorkerActor(ToolWorkerConfig(cwd=str(tmp_path))).Read(file_path=str(p))
    assert out["content"] == "hello"
    assert out.get("is_error") is False


def test_tool_worker_actor_default_config_no_on_start() -> None:
    """Constructing without a config (and without on_start) must not crash."""
    from pulsing.forge import ToolWorkerActor

    out = ToolWorkerActor().Glob(pattern="*.md")
    assert out.get("is_error") is False


def test_tool_result_dataclass() -> None:
    pytest.importorskip("anthropic")
    from pulsing.agent.loop.tool_base import ToolResult

    r = ToolResult(content="ok")
    assert r.content == "ok"
    assert r.is_error is False


def test_tool_result_from_worker_value() -> None:
    from pulsing.agent.loop.tool_base import tool_result_from_worker_value

    r = tool_result_from_worker_value({"content": "x", "is_error": True})
    assert r.content == "x" and r.is_error is True


def test_split_tools_builds_core_list() -> None:
    from pulsing.agent.loop.forge_tools import assert_forge_tool_coverage
    from pulsing.agent.loop.permissions import PermissionChecker
    from pulsing.agent.loop.split_tools import build_tools_for_agent
    from pulsing.forge.integrated import FORGE_TOOL_NAMES

    assert_forge_tool_coverage()
    tools = build_tools_for_agent(PermissionChecker(auto_approve=True))
    names = {t.name for t in tools}
    assert FORGE_TOOL_NAMES <= names
    assert "FetchUrl" in names
    assert "QuestReport" in names
    with pytest.raises(RuntimeError):
        next(t for t in tools if t.name == "Read").execute(file_path=".")


def test_split_tools_includes_cluster_when_enabled() -> None:
    from pulsing.agent.loop.permissions import PermissionChecker
    from pulsing.agent.loop.split_tools import build_tools_for_agent

    tools = build_tools_for_agent(
        PermissionChecker(auto_approve=True),
        cluster_enabled=True,
        summon_enabled=True,
    )
    names = {t.name for t in tools}
    assert "ListClusterAgents" in names
    assert "MessageClusterAgent" in names
    assert "Summon" in names
    assert "QuestReport" in names
