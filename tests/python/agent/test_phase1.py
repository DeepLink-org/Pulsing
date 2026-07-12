# SPDX-License-Identifier: Apache-2.0
"""Phase 1: workspace/cluster/npc live in pulsing.agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.agent import AgentConfig, find_workspace_root, list_cluster_agents
from pulsing.agent.cluster.constants import full_agent_name, workspace_agent_name
from pulsing.agent.npc import list_npc_classes, seed_npc_defs
from pulsing.agent.workspace.config import default_config, save_config


def test_agent_workspace_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    cfg = default_config(root)
    save_config(cfg)
    found = find_workspace_root(root)
    assert found == root.resolve()
    assert cfg.cluster_id == default_config(root).cluster_id


def test_workspace_load_config(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    cfg = default_config(root)
    save_config(cfg)
    from pulsing.agent.workspace.config import load_config

    loaded = load_config(root)
    assert cfg.cluster_id == loaded.cluster_id


def test_agent_config_alias() -> None:
    assert AgentConfig is not None
    cfg = AgentConfig(
        model="x",
        cwd="/tmp",
        agent_name="guide",
        workspace_id="abc123",
    )
    assert cfg.short_name == "guide"


def test_workspace_agent_naming() -> None:
    name = workspace_agent_name("ws1", "coder")
    assert name == full_agent_name("coder", workspace_id="ws1")
    assert name.startswith("craft/ws/")


def test_npc_classes_builtin() -> None:
    names = list_npc_classes()
    assert "artisan" in names


def test_seed_npc_defs(tmp_path: Path) -> None:
    seed_npc_defs(tmp_path)
    assert (tmp_path / ".pulsing" / "npcs" / "artisan.json").is_file()


@pytest.mark.asyncio
async def test_list_cluster_agents_empty_when_no_system() -> None:
    """Import path smoke; full cluster tests live in tests/python/agent/."""
    assert callable(list_cluster_agents)
