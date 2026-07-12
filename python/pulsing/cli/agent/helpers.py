# SPDX-License-Identifier: Apache-2.0
"""Shared ``pulsing agent`` CLI helpers."""

from __future__ import annotations

import os
from typing import Any

import pulsing as pul

from pulsing.agent.cluster.constants import full_agent_name
from pulsing.agent.cluster.discovery import list_cluster_agents
from pulsing.agent.npc import spawn_npc as _spawn_npc
from pulsing.agent.npc.config import NpcConfig
from pulsing.agent.loop.deps import require_provider_deps
from pulsing.agent.workspace.config import WorkspaceConfig, load_config
from pulsing.agent.workspace.root import require_workspace_root
from pulsing.agent.workspace.session import workspace_session

DEFAULT_PROG = "pulsing agent"


def default_model(provider: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if (provider or "anthropic").strip().lower() == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o")
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def load_cfg() -> WorkspaceConfig:
    return load_config(require_workspace_root())


def llm_options(cfg: WorkspaceConfig, args: Any) -> dict[str, Any]:
    provider = getattr(args, "provider", None) or cfg.provider
    require_provider_deps(provider)
    return {
        "provider": provider,
        "model": default_model(provider, getattr(args, "model", None) or cfg.model),
        "auto_approve": bool(getattr(args, "auto_approve", False) or cfg.auto_approve),
        "sandbox": cfg.sandbox,
    }


async def spawn_npc(
    cfg: WorkspaceConfig,
    name: str,
    llm: dict[str, Any],
    *,
    role: str = "",
    npc_class: str = "artisan",
    shared_tool_worker: bool | None = None,
) -> Any:
    config = NpcConfig(
        model=llm["model"],
        cwd=cfg.root,
        agent_name=name,
        workspace_id=cfg.cluster_id,
        provider=llm["provider"],
        auto_approve=llm["auto_approve"],
        sandbox_policy=llm["sandbox"],
        agent_role=role,
        npc_class=npc_class,
        shared_tool_worker=cfg.shared_tool_worker
        if shared_tool_worker is None
        else shared_tool_worker,
    )
    return await _spawn_npc(
        config,
        name=full_agent_name(name, workspace_id=cfg.cluster_id),
        public=True,
    )


async def with_session(cfg: WorkspaceConfig, fn: Any) -> None:
    async with workspace_session(cfg):
        await fn()


async def list_npcs(cfg: WorkspaceConfig) -> list[dict[str, Any]] | None:
    if not cfg.seed_addr():
        return None
    async with workspace_session(cfg):
        return await list_cluster_agents(pul.get_system(), workspace_id=cfg.cluster_id)
