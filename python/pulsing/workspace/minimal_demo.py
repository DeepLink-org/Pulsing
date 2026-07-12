# SPDX-License-Identifier: Apache-2.0
"""Minimal workspace demo: init → wake → say in one process."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Literal

import pulsing as pul

from pulsing.agent.cluster.resolve import resolve_agent
from pulsing.agent.workspace.config import load_config, save_config, write_node_record
from pulsing.agent.workspace.session import workspace_session
from pulsing.agent.workspace.world_view import player_name
from pulsing.cli.agent.helpers import spawn_npc
from pulsing.forge.host.llm import llm_runtime_options
from pulsing.workspace.bootstrap import init_workspace

Template = Literal["agent", "minimal"]


async def run_workspace_minimal_demo(
    root: Path,
    *,
    message: str = "list project files with Glob",
    provider: str = "demo",
    model: str | None = None,
    template: Template = "agent",
) -> dict[str, Any]:
    """Bootstrap workspace, spawn guide, deliver one message, return agent output."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    init_workspace(root, template=template, seed_npcs=(template == "agent"))
    cfg = load_config(root)
    cfg.provider = provider
    cfg.model = model or ("demo" if provider == "demo" else cfg.model)
    cfg.auto_approve = True
    save_config(cfg)

    llm = llm_runtime_options(
        provider=provider,
        model=cfg.model,
        auto_approve=True,
        sandbox=cfg.sandbox,
    )

    async with workspace_session(cfg, bind_addr="127.0.0.1:0") as system:
        write_node_record(cfg, addr=str(system.addr), pid=os.getpid())
        await spawn_npc(cfg, "guide", llm, role="guide")
        proxy = await _wait_for_agent(cfg, "guide")
        out = await proxy.deliver_message(
            from_sender=player_name(),
            message=message,
            channel="say",
        )
    if not isinstance(out, dict):
        return {"assistant_text": str(out)}
    return out


async def _wait_for_agent(cfg, name: str, *, timeout: float = 30.0):
    deadline = asyncio.get_running_loop().time() + timeout
    last_err: Exception | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            return await resolve_agent(
                pul.get_system(),
                name,
                workspace_id=cfg.cluster_id,
                timeout=5.0,
            )
        except Exception as exc:
            last_err = exc
            await asyncio.sleep(0.2)
    raise RuntimeError(f"agent {name!r} not ready") from last_err
