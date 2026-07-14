# SPDX-License-Identifier: Apache-2.0
"""Workspace commands: init, wake, look, sleep."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from argparse import Namespace
from pathlib import Path

from pulsing.agent.workspace.config import (
    clear_node_record,
    load_config,
    save_config,
)
from pulsing.agent.workspace.config import write_node_record
from pulsing.agent.workspace.root import find_workspace_root
from pulsing.agent.workspace.session import workspace_session
from pulsing.agent.workspace.tool_pool import spawn_shared_tool_worker
from pulsing.agent.workspace.world_view import render_look
from pulsing.cli.agent.helpers import DEFAULT_PROG, list_npcs, llm_options, spawn_npc


async def run_init(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    from pulsing.workspace.bootstrap import init_workspace

    root = Path.cwd().resolve()
    template = getattr(args, "template", "agent") or "agent"
    force = bool(getattr(args, "force", False))
    guide_flag = getattr(args, "guide", None)
    guide_words = getattr(args, "guide_words", None) or []
    if guide_flag and str(guide_flag).strip():
        guide = str(guide_flag).strip()
    elif guide_words:
        guide = " ".join(guide_words)
    else:
        guide = None
    result = init_workspace(
        root,
        template=template,
        force=force,
        guide=guide,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
    )
    if result.created:
        print(f"initialized {result.root}  →  {prog} wake  ·  {prog} dashboard")
    else:
        print(f"already initialized: {result.root}")


async def run_look(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    root = find_workspace_root()
    if not root:
        print(f"no workspace — run `{prog} init`")
        return

    async def produce() -> str:
        cfg = load_config(root)
        rows = await list_npcs(cfg)
        return render_look(cfg, npc_rows=rows)

    if not getattr(args, "follow", False):
        print(await produce())
        return

    from pulsing.cli.agent.commands.follow import follow_output

    interval = max(0.5, float(getattr(args, "interval", 5.0)))
    try:
        await follow_output(
            produce,
            interval=interval,
            scroll=True,
            delta=not getattr(args, "no_delta", False),
        )
    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        print("\n(stopped)")


async def run_wake(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    from pulsing.cli.agent.helpers import load_cfg

    cfg = load_cfg()
    llm = llm_options(cfg, args)
    shared = bool(getattr(args, "shared_tool_worker", False) or cfg.shared_tool_worker)
    if shared and not cfg.shared_tool_worker:
        cfg.shared_tool_worker = True
        save_config(cfg)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop.set())
    async with workspace_session(cfg, bind_addr=args.addr) as system:
        write_node_record(cfg, addr=str(system.addr), pid=os.getpid())
        if shared:
            await spawn_shared_tool_worker(cfg)
        for part in (args.agents or ",".join(cfg.default_agents)).split(","):
            name = part.strip()
            if name:
                await spawn_npc(cfg, name, llm, shared_tool_worker=shared)
        print(f"awake at {system.addr} — try: {prog} dashboard", file=sys.stderr)
        await stop.wait()
    clear_node_record(cfg)


async def run_sleep(_args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    raise SystemExit(f"{prog} sleep is not implemented yet")
