# SPDX-License-Identifier: Apache-2.0
"""Tail one agent's in-memory log stream."""

from __future__ import annotations

import asyncio
from argparse import Namespace

import pulsing as pul

from pulsing.agent.cluster.resolve import resolve_agent
from pulsing.cli.agent.helpers import DEFAULT_PROG, load_cfg, with_session


async def run_logs(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    name = (args.name or "").strip()
    if not name:
        raise SystemExit(f"usage: {prog} logs <name> [-f]")
    cfg = load_cfg()
    follow = bool(getattr(args, "follow", False))
    interval = max(0.2, float(getattr(args, "interval", 0.8)))
    since = max(0, int(getattr(args, "since", 0)))

    async def _tail_once() -> int:
        proxy = await resolve_agent(
            pul.get_system(),
            name,
            workspace_id=cfg.cluster_id,
            timeout=30.0,
        )
        chunk = await proxy.get_logs(since=since)
        if not isinstance(chunk, dict):
            return since
        for line in chunk.get("lines") or []:
            print(line, flush=True)
        return int(chunk.get("next") or since)

    async def _go() -> None:
        nonlocal since
        if not follow:
            since = await _tail_once()
            return
        print(f"── {name} log (follow) ──", flush=True)
        while True:
            since = await _tail_once()
            await asyncio.sleep(interval)

    try:
        await with_session(cfg, _go)
    except KeyboardInterrupt:
        print("\n(stopped)")
