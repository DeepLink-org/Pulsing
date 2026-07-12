# SPDX-License-Identifier: Apache-2.0
"""Puzzle commands: list, show, mark."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path

from pulsing.agent.workspace.config import WorkspaceConfig, load_config, save_config
from pulsing.agent.workspace.quest import (
    QUEST_STATUSES,
    normalize_quest,
    update_quest_status,
)
from pulsing.agent.workspace.world_view import format_puzzles
from pulsing.cli.agent.helpers import DEFAULT_PROG, load_cfg


def _format_one(cfg: WorkspaceConfig, pid: str) -> str:
    puzzle = normalize_quest(cfg.puzzles.get(pid, {}))
    if pid not in cfg.puzzles:
        raise SystemExit(f"unknown puzzle {pid!r}")
    kind = puzzle.get("kind") or "task"
    title = puzzle.get("title") or pid
    path = puzzle.get("path") or "."
    status = puzzle.get("status") or "open"
    assign = puzzle.get("assign_to") or ""
    lines = [f"{pid} [{kind}/{status}] {title} @ {path}"]
    if assign:
        lines.append(f"  assign_to: {assign}")
    if puzzle.get("blurb"):
        lines.append(f"  {puzzle['blurb']}")
    if puzzle.get("last_note"):
        lines.append(f"  note: {puzzle['last_note']}")
    return "\n".join(lines)


async def run_list(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    if not getattr(args, "follow", False):
        print(format_puzzles(load_cfg(), all_=True))
        return

    from pulsing.cli.agent.commands.follow import follow_output

    interval = max(0.5, float(getattr(args, "interval", 5.0)))

    async def produce() -> str:
        return format_puzzles(load_cfg(), all_=True)

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


async def run_show(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    print(_format_one(load_cfg(), args.id))


async def run_mark(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    cfg = load_cfg()
    pid = args.id
    if pid not in cfg.puzzles:
        raise SystemExit(f"unknown puzzle {pid!r}")
    status = args.status
    if status not in QUEST_STATUSES:
        raise SystemExit(f"invalid status {status!r}")
    root = Path(cfg.root)
    updated = update_quest_status(root, pid, status=status, reporter="player")
    if args.assign_to is not None:
        cfg = load_config(root)
        p = normalize_quest(cfg.puzzles[pid])
        p["assign_to"] = args.assign_to.strip()
        cfg.puzzles[pid] = p
        save_config(cfg)
        updated = p
    print(_format_one(load_config(root), pid))
