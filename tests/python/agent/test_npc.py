# SPDX-License-Identifier: Apache-2.0
"""Workspace NPCs: NpcConfig, spawn_npc."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from pulsing.agent.actors.npc import format_incoming
from pulsing.agent.cluster.constants import full_agent_name, is_public_npc_name
from pulsing.agent.cluster.discovery import list_cluster_agents
from pulsing.agent.npc import spawn_npc, get_npc_class, list_npc_classes, seed_npc_defs
from pulsing.agent.npc.config import NpcConfig
from pulsing.agent.workspace.config import default_config, save_config


def test_format_incoming_whisper() -> None:
    assert format_incoming("alice", "hi", channel="whisper") == "[alice whispers]\nhi"
    assert format_incoming("bob", "hey", channel="say") == "[bob]\nhey"


def test_npc_config_resolved_profile(tmp_path: Path) -> None:
    cfg = NpcConfig(
        model="m",
        cwd=str(tmp_path),
        agent_name="guide",
        workspace_id="ws1",
        npc_class="scholar",
    )
    prompt, allow, forbid, cls_name, _ = cfg.resolved_profile()
    assert "guide" in prompt
    assert "scholar" == cls_name
    assert "Read" in allow
    assert "Edit" in forbid


def test_npc_class_registry() -> None:
    assert "artisan" in list_npc_classes()
    quest = get_npc_class("quest_giver")
    assert "Summon" in quest.default_tools
    assert "MessageClusterAgent" in quest.default_tools
    assert "QuestReport" in quest.default_tools
    assert "Edit" in get_npc_class("scholar").forbidden_tools


def test_seed_and_load_external_npc(tmp_path: Path) -> None:
    seed_npc_defs(tmp_path)
    nd = tmp_path / ".pulsing" / "npcs"
    assert (nd / "artisan.json").is_file()
    custom = {
        "name": "ranger",
        "description": "scout",
        "default_tools": ["Read", "Glob"],
        "forbidden_tools": ["Bash"],
    }
    (nd / "ranger.json").write_text(json.dumps(custom), encoding="utf-8")
    cls = get_npc_class("ranger", tmp_path)
    assert cls.description == "scout"
    assert "ranger" in list_npc_classes(tmp_path)


def test_is_public_npc_name_filters_internals() -> None:
    assert is_public_npc_name("guide") is True
    assert is_public_npc_name("_engine") is False
    assert is_public_npc_name("guide/_engine") is False
    assert is_public_npc_name("guide/tasks/task-1") is False


@pytest.mark.asyncio
async def test_spawn_ping_and_cluster_info(tmp_path: Path) -> None:
    import pulsing as pul

    cfg = default_config(tmp_path)
    save_config(cfg)
    ws = cfg.cluster_id
    name = f"npc-{uuid.uuid4().hex[:6]}"

    await pul.init()
    try:
        config = NpcConfig(
            model="claude-sonnet-4-20250514",
            cwd=str(tmp_path),
            agent_name=name,
            workspace_id=ws,
            auto_approve=True,
            agent_role="guide",
        )
        npc = await spawn_npc(
            config,
            name=full_agent_name(name, workspace_id=ws),
            public=True,
        )
        ping = await npc.ping()
        assert ping == {"ok": True, "kind": "npc", "name": name}

        info = await npc.get_cluster_info()
        assert info["full_name"] == f"craft/ws/{ws}/{name}"
        assert info["kind"] == "npc"
        assert await npc.metadata() == {
            "agent.kind": "workspace",
            "agent.name": name,
            "agent.class": "artisan",
            "agent.role": "guide",
            "agent.workspace_id": ws,
        }

        rows = await list_cluster_agents(pul.get_system(), workspace_id=ws)
        names = {r["name"] for r in rows}
        assert name in names
        assert not any("/" in n or n.startswith("_") for n in names)
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_deliver_message_wait_false_mailbox_handoff(tmp_path: Path) -> None:
    import pulsing as pul

    cfg = default_config(tmp_path)
    save_config(cfg)
    ws = cfg.cluster_id
    name = f"q-{uuid.uuid4().hex[:6]}"

    await pul.init()
    try:
        config = NpcConfig(
            model="claude-sonnet-4-20250514",
            cwd=str(tmp_path),
            agent_name=name,
            workspace_id=ws,
            auto_approve=True,
        )
        npc = await spawn_npc(
            config,
            name=full_agent_name(name, workspace_id=ws),
            public=True,
        )
        out = await npc.deliver_message(
            from_sender="player",
            message="hello",
            channel="say",
            wait=False,
        )
        assert out.get("ok") is True
        assert out.get("accepted") is True
    finally:
        await pul.shutdown()
