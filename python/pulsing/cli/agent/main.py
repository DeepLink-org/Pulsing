# SPDX-License-Identifier: Apache-2.0
"""Agent CLI: ``pulsing agent``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from argparse import Namespace
from collections.abc import Sequence

from pulsing.cli.agent.commands import (
    agent_cmd,
    dashboard,
    demo,
    npc,
    puzzle,
    watch,
    world,
)

DEFAULT_PROG = "pulsing agent"


def _build_parser(prog: str = DEFAULT_PROG) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Agent workspace CLI (init, wake, spawn, task).",
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("init", help="create .pulsing/ workspace")

    w = sub.add_parser("wake", help="start node and run NPCs (blocking)")
    w.add_argument("--agents", default=None)
    w.add_argument("--addr", default="127.0.0.1:0")
    w.add_argument("--auto-approve", action="store_true")
    w.add_argument("--provider", choices=("anthropic", "openai"))
    w.add_argument("--model")
    w.add_argument(
        "--shared-tool-worker",
        action="store_true",
        help="one isolated worker for all NPCs",
    )

    look_p = sub.add_parser("look", help="show workspace summary (default command)")
    look_p.add_argument(
        "-f", "--follow", action="store_true", help="append snapshots (scroll log)"
    )
    look_p.add_argument("-i", "--interval", type=float, default=5.0)
    look_p.add_argument(
        "--no-delta", action="store_true", help="print every tick even if unchanged"
    )

    sub.add_parser("list", help="list agent names in the cluster")

    s = sub.add_parser("spawn", help="spawn an agent")
    s.add_argument("name")
    s.add_argument("--role", default="")
    s.add_argument("--provider", choices=("anthropic", "openai"))
    s.add_argument("--model")

    y = sub.add_parser("say", help="send a message to an agent")
    y.add_argument("name")
    y.add_argument("message", nargs=argparse.REMAINDER)

    sub.add_parser("sleep", help="snapshot and stop node (not implemented)")

    logs_p = sub.add_parser("logs", help="tail one agent's in-memory log")
    logs_p.add_argument("name", help="agent short name (e.g. guide)")
    logs_p.add_argument(
        "-f", "--follow", action="store_true", help="stream until Ctrl+C"
    )
    logs_p.add_argument(
        "-i", "--interval", type=float, default=0.8, help="poll interval with --follow"
    )
    logs_p.add_argument(
        "--since", type=int, default=0, help="start after log sequence id"
    )

    dash = sub.add_parser(
        "dashboard",
        help="split-terminal UI (Zellij/tmux): one pane per agent log + player shell",
    )
    dash.add_argument(
        "--backend",
        choices=("auto", "zellij", "tmux"),
        default="auto",
        help="auto prefers Zellij (Rust multiplexer)",
    )
    dash.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.8,
        help="agent log poll interval (seconds)",
    )

    wch = sub.add_parser("watch", help="show what each agent is doing in the cluster")
    wch.add_argument("-f", "--follow", action="store_true", help="refresh until Ctrl+C")
    wch.add_argument(
        "--scroll", action="store_true", help="append updates (no full-screen clear)"
    )
    wch.add_argument(
        "--no-delta", action="store_true", help="print every tick even if unchanged"
    )
    wch.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2.0,
        help="refresh seconds (with --follow)",
    )
    wch.add_argument("--local", action="store_true", help="only agents on this node")

    demo_p = sub.add_parser(
        "demo",
        help="one-shot: 3 chattering demo agents + optional dashboard (no API key by default)",
    )
    demo_p.add_argument("--addr", default="127.0.0.1:0")
    demo_p.add_argument(
        "--interval", type=float, default=6.0, help="seconds between chatter messages"
    )
    demo_p.add_argument(
        "--no-dashboard", action="store_true", help="do not spawn dashboard"
    )
    demo_p.add_argument(
        "--real-llm",
        action="store_true",
        help="use configured LLM provider instead of offline demo LLM",
    )

    task_p = sub.add_parser("task", help="task commands (alias for puzzle)")
    task_sub = task_p.add_subparsers(dest="task_cmd")
    plist = task_sub.add_parser("list", help="list tasks")
    plist.add_argument(
        "-f", "--follow", action="store_true", help="append snapshots (scroll log)"
    )
    plist.add_argument("-i", "--interval", type=float, default=5.0)
    plist.add_argument("--no-delta", action="store_true")
    show = task_sub.add_parser("show", help="show one task")
    show.add_argument("id")
    mark = task_sub.add_parser("mark", help="update task status (not implemented)")
    mark.add_argument("id")
    mark.add_argument(
        "--status", choices=("open", "in_progress", "solved", "failed"), required=True
    )
    mark.add_argument("--assign-to", default=None, help="assign task to agent name")

    return p


def _parse(argv: Sequence[str], *, prog: str = DEFAULT_PROG) -> Namespace:
    args = _build_parser(prog=prog).parse_args(list(argv))
    if args.cmd is None:
        args.cmd = "look"
    if args.cmd == "task" and args.task_cmd is None:
        raise SystemExit(f"usage: {prog} task list|show|mark …")
    return args


async def _dispatch(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    cmd = args.cmd

    if cmd == "init":
        await world.run_init(args, prog=prog)
        return
    if cmd == "look":
        await world.run_look(args, prog=prog)
        return
    if cmd == "watch":
        await watch.run_watch(args, prog=prog)
        return
    if cmd == "dashboard":
        dashboard.run_dashboard(args, prog=prog)
        return
    if cmd == "logs":
        await agent_cmd.run_logs(args, prog=prog)
        return
    if cmd == "demo":
        await demo.run_demo(args, prog=prog)
        return
    if cmd == "wake":
        await world.run_wake(args, prog=prog)
        return
    if cmd == "sleep":
        await world.run_sleep(args, prog=prog)
        return
    if cmd == "list":
        await npc.run_who(args, prog=prog)
        return
    if cmd == "spawn":
        await npc.run_summon(args, prog=prog)
        return
    if cmd == "say":
        await npc.run_say(args, prog=prog)
        return
    if cmd == "task":
        if args.task_cmd == "list":
            await puzzle.run_list(args, prog=prog)
            return
        if args.task_cmd == "show":
            await puzzle.run_show(args, prog=prog)
            return
        if args.task_cmd == "mark":
            await puzzle.run_mark(args, prog=prog)
            return
        raise SystemExit(f"usage: {prog} task list|show|mark …")

    raise SystemExit(f"unknown command {cmd!r}")


async def async_main(
    argv: Sequence[str] | None = None, *, prog: str = DEFAULT_PROG
) -> None:
    tokens = list(argv) if argv is not None else []
    args = _parse(tokens, prog=prog)
    await _dispatch(args, prog=prog)


def main(argv: Sequence[str] | None = None, *, prog: str = DEFAULT_PROG) -> None:
    if argv is None:
        argv = sys.argv[1:]
    try:
        asyncio.run(async_main(argv, prog=prog))
    except KeyboardInterrupt:
        print("\n(interrupted)", file=sys.stderr)


if __name__ == "__main__":
    main()
