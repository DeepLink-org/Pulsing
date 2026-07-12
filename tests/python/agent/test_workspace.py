# SPDX-License-Identifier: Apache-2.0
"""Workspace config, naming, scoped discovery."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from pulsing.agent.cluster.constants import (
    full_agent_name,
    short_agent_name,
    workspace_agent_name,
)
from pulsing.agent.cluster.discovery import list_cluster_agents
from pulsing.agent.npc import spawn_npc
from pulsing.agent.npc.config import NpcConfig
from pulsing.agent.workspace.config import (
    default_config,
    load_config,
    save_config,
    write_node_record,
)
from pulsing.agent.workspace.root import find_workspace_root, workspace_cluster_id


def test_workspace_agent_naming() -> None:
    ws = "abc123"
    assert workspace_agent_name(ws, "lead") == "craft/ws/abc123/lead"
    assert short_agent_name("craft/ws/abc123/coder", workspace_id=ws) == "coder"
    assert full_agent_name("lead", workspace_id=ws) == "craft/ws/abc123/lead"


def test_init_and_find_root(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    save_config(cfg)
    assert find_workspace_root(tmp_path / "src" / "nested") == tmp_path.resolve()
    assert load_config(tmp_path).cluster_id == workspace_cluster_id(tmp_path)


def test_node_record_roundtrip(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    save_config(cfg)
    write_node_record(cfg, addr="127.0.0.1:9001", pid=42)
    data = json.loads(cfg.node_path.read_text())
    assert data["addr"] == "127.0.0.1:9001"
    assert cfg.seed_addr() == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_scoped_discovery_isolates_workspace(tmp_path: Path) -> None:
    import pulsing as pul

    cfg = default_config(tmp_path)
    save_config(cfg)
    ws = cfg.cluster_id
    name = f"lead-{uuid.uuid4().hex[:6]}"

    await pul.init()
    try:
        config = NpcConfig(
            model="claude-sonnet-4-20250514",
            cwd=str(tmp_path),
            agent_name=name,
            workspace_id=ws,
            auto_approve=True,
        )
        await spawn_npc(
            config,
            name=full_agent_name(name, workspace_id=ws),
            public=True,
        )
        rows = await list_cluster_agents(pul.get_system(), workspace_id=ws)
        assert any(r["name"] == name for r in rows)
        other = await list_cluster_agents(
            pul.get_system(), workspace_id="other-workspace-id"
        )
        assert not any(r["name"] == name for r in other)
    finally:
        await pul.shutdown()
