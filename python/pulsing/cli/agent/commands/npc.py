# SPDX-License-Identifier: Apache-2.0
"""NPC commands: who, summon, say."""

from __future__ import annotations

from argparse import Namespace

import pulsing as pul

from pulsing.agent.cluster.discovery import list_cluster_agents
from pulsing.agent.cluster.resolve import resolve_agent
from pulsing.agent.workspace.world_view import player_name
from pulsing.cli.agent.helpers import (
    DEFAULT_PROG,
    llm_options,
    load_cfg,
    spawn_npc,
    with_session,
)


async def run_who(_args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    cfg = load_cfg()

    async def _go() -> None:
        rows = await list_cluster_agents(pul.get_system(), workspace_id=cfg.cluster_id)
        if not rows:
            print("(no NPCs)")
            return
        for row in rows:
            print(row.get("name", "?"))

    await with_session(cfg, _go)


async def run_summon(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    cfg = load_cfg()
    llm = llm_options(cfg, args)

    async def _go() -> None:
        await spawn_npc(cfg, args.name, llm, role=args.role)

    await with_session(cfg, _go)
    print(f"summoned {args.name}")


async def run_say(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    cfg = load_cfg()
    msg = " ".join(args.message or []).strip()
    if not msg:
        raise SystemExit(f"usage: {prog} npc say <npc> message…")

    async def _go() -> None:
        proxy = await resolve_agent(
            pul.get_system(),
            args.name,
            workspace_id=cfg.cluster_id,
        )
        out = await proxy.deliver_message(
            from_sender=player_name(),
            message=msg,
            channel="say",
        )
        body = (
            out.get("assistant_text") or out.get("error")
            if isinstance(out, dict)
            else out
        )
        print(f"\n{args.name} › {body}")

    await with_session(cfg, _go)
