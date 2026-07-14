# SPDX-License-Identifier: Apache-2.0
"""Cluster naming and gossip discovery."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from pulsing.agent.cluster.constants import full_agent_name, short_agent_name
from pulsing.agent.cluster.discovery import format_agent_table, list_cluster_agents
from pulsing.agent.npc.config import NpcConfig
from pulsing.agent.workspace.config import default_config, save_config


def test_agent_name_roundtrip() -> None:
    ws = "abc123"
    assert full_agent_name("coder", workspace_id=ws) == "craft/ws/abc123/coder"
    assert short_agent_name("craft/ws/abc123/coder", workspace_id=ws) == "coder"
    assert short_agent_name("coder") == "coder"


def test_format_agent_table_empty() -> None:
    assert "no cluster agents" in format_agent_table([]).lower()


@pytest.mark.asyncio
async def test_cluster_info_and_ping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pulsing as pul
    from pulsing.agent.actors import Agent

    monkeypatch.chdir(tmp_path)
    cfg = default_config(tmp_path)
    save_config(cfg)
    ws = cfg.cluster_id

    await pul.init()
    try:
        name = f"t-{uuid.uuid4().hex[:8]}"
        config = NpcConfig(
            model="claude-sonnet-4-20250514",
            cwd=str(tmp_path),
            agent_name=name,
            workspace_id=ws,
            auto_approve=True,
            agent_role="tester",
            agent_description="smoke",
        )
        agent = await Agent.spawn(
            config=config,
            name=full_agent_name(name, workspace_id=ws),
            public=True,
        )
        info = await agent.get_cluster_info()
        assert info["name"] == name
        assert info["role"] == "tester"
        assert info["cluster_enabled"] is True
        ping = await agent.ping()
        assert ping["ok"] is True
        assert ping["name"] == name
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_get_activity_idle(tmp_path: Path) -> None:
    import asyncio

    import pulsing as pul
    from pulsing.agent.actors import Agent
    from pulsing.agent.workspace.config import default_config, save_config

    cfg = default_config(tmp_path)
    save_config(cfg)
    ws = cfg.cluster_id
    name = f"act-{__import__('uuid').uuid4().hex[:6]}"

    await pul.init()
    try:
        from pulsing.agent.npc.config import NpcConfig

        config = NpcConfig(
            model="claude-sonnet-4-20250514",
            cwd=str(tmp_path),
            agent_name=name,
            workspace_id=ws,
            auto_approve=True,
        )
        agent = await Agent.spawn(
            config=config,
            name=f"craft/ws/{ws}/{name}",
            public=True,
        )
        await asyncio.sleep(0.3)
        act = await agent.get_activity()
        assert act.get("state") == "idle"
        assert act.get("name") == name
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_list_cluster_agents_finds_spawned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pulsing as pul
    from pulsing.agent.actors import Agent

    monkeypatch.chdir(tmp_path)
    cfg = default_config(tmp_path)
    save_config(cfg)
    ws = cfg.cluster_id

    await pul.init()
    try:
        name = f"disc-{uuid.uuid4().hex[:8]}"
        config = NpcConfig(
            model="claude-sonnet-4-20250514",
            cwd=str(tmp_path),
            agent_name=name,
            workspace_id=ws,
            auto_approve=True,
        )
        await Agent.spawn(
            config=config,
            name=full_agent_name(name, workspace_id=ws),
            public=True,
        )
        rows = await list_cluster_agents(pul.get_system(), workspace_id=ws)
        assert name in {r["name"] for r in rows}
    finally:
        await pul.shutdown()
