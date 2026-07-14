# SPDX-License-Identifier: Apache-2.0
"""Watch cluster NPC activity."""

from __future__ import annotations

import asyncio
import sys
from argparse import Namespace
from datetime import datetime

import pulsing as pul

from pulsing.agent.cluster.activity import (
    collect_cluster_activity,
    format_activity_table,
)
from pulsing.cli.agent.commands.follow import follow_output
from pulsing.cli.agent.helpers import DEFAULT_PROG, load_cfg, with_session


async def _activity_text(args: Namespace) -> str:
    cfg = load_cfg()
    buf = ""

    async def _go() -> None:
        nonlocal buf
        rows = await collect_cluster_activity(
            pul.get_system(),
            workspace_id=cfg.cluster_id,
            local_node_only=bool(getattr(args, "local", False)),
        )
        title = f"{cfg.name} activity"
        buf = format_activity_table(rows, title=title)

    await with_session(cfg, _go)
    return buf


async def run_watch(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    if not getattr(args, "follow", False):
        print(await _activity_text(args))
        return

    interval = max(0.5, float(getattr(args, "interval", 2.0)))
    scroll = bool(getattr(args, "scroll", False))
    delta = not getattr(args, "no_delta", False)

    try:
        if scroll:
            await follow_output(
                lambda: _activity_text(args),
                interval=interval,
                scroll=True,
                delta=delta,
            )
            return
        while True:
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"{load_cfg().name} @ {stamp}")
            print(await _activity_text(args))
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        print("\n(stopped)")
