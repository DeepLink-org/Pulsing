# SPDX-License-Identifier: Apache-2.0
"""One-shot demo: three chattering NPCs + optional Zellij split UI."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import signal
import sys
from argparse import Namespace
from pathlib import Path

import pulsing as pul

from pulsing.agent.cluster.resolve import resolve_agent
from pulsing.cli.agent.commands.dashboard import (
    DEMO_WORKER_ENV,
    ZELLIJ_DEMO_SESSION,
    agent_cli_argv,
    launch_zellij,
    write_zellij_demo_layout,
)
from pulsing.cli.agent.helpers import DEFAULT_PROG, spawn_npc
from pulsing.agent.npc.loader import seed_npc_defs
from pulsing.agent.workspace.config import (
    WorkspaceConfig,
    clear_node_record,
    default_config,
    load_config,
    save_config,
    write_node_record,
)
from pulsing.agent.workspace.root import find_workspace_root
from pulsing.agent.workspace.session import workspace_session
from pulsing.agent.workspace.tool_pool import spawn_shared_tool_worker

DEMO_AGENTS: list[tuple[str, str, str]] = [
    ("bard", "quest_giver", "storyteller"),
    ("smith", "artisan", "builder"),
    ("sage", "scholar", "reviewer"),
]

CHATTER_SCRIPT: list[tuple[str, str, str]] = [
    ("bard", "smith", "List project files with Glob under ."),
    ("smith", "sage", "Glob under tests — one line summary."),
    ("sage", "bard", "Update unit-tests quest — use QuestReport in_progress."),
    ("bard", "smith", "MessageClusterAgent to sage: coordinate on tests path."),
    ("smith", "bard", "Glob the tests directory and report."),
    ("sage", "smith", "Peer review: any blockers for unit-tests quest?"),
]


def _demo_llm_options(cfg: WorkspaceConfig, args: Namespace) -> dict:
    if getattr(args, "real_llm", False):
        from pulsing.cli.agent.helpers import llm_options

        return llm_options(cfg, args)
    from pulsing.forge.host.llm import llm_runtime_options

    return llm_runtime_options(
        provider="demo",
        model="demo",
        auto_approve=True,
        sandbox=cfg.sandbox,
    )


def prepare_demo_workspace(root: Path, *, prog: str = DEFAULT_PROG) -> WorkspaceConfig:
    root = root.resolve()
    existing = find_workspace_root(root)
    if existing:
        cfg = load_config(existing)
    else:
        cfg = default_config(root)
    seed_npc_defs(root)
    cfg.default_agents = [a[0] for a in DEMO_AGENTS]
    cfg.shared_tool_worker = True
    cfg.auto_approve = True
    puzzles = dict(cfg.puzzles)
    unit = dict(puzzles.get("unit-tests") or {})
    unit["assign_to"] = "sage"
    unit["status"] = "open"
    puzzles["unit-tests"] = unit
    cfg.puzzles = puzzles
    save_config(cfg)
    return cfg


def demo_worker_shell(args: Namespace, *, prog: str = DEFAULT_PROG) -> str:
    argv = list(agent_cli_argv("demo", "--no-dashboard"))
    if getattr(args, "real_llm", False):
        argv.append("--real-llm")
    argv.extend(["--interval", str(float(getattr(args, "interval", 6.0)))])
    addr = getattr(args, "addr", None)
    if addr:
        argv.extend(["--addr", str(addr)])
    cmd = " ".join(shlex.quote(p) for p in argv)
    return f"export {DEMO_WORKER_ENV}=1; {cmd}"


def try_exec_demo_zellij(
    cfg: WorkspaceConfig,
    args: Namespace,
    *,
    prog: str = DEFAULT_PROG,
    log_interval: float = 0.8,
) -> None:
    if getattr(args, "no_dashboard", False):
        return
    if os.environ.get(DEMO_WORKER_ENV):
        return
    if not shutil.which("zellij"):
        print(
            f"tip: install zellij for split-screen — brew install zellij; or `{prog} agent logs bard -f`",
            file=sys.stderr,
        )
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            f"non-interactive shell — run `{prog} agent logs bard -f` in another terminal",
            file=sys.stderr,
        )
        return
    layout = write_zellij_demo_layout(
        cfg,
        demo_shell=demo_worker_shell(args, prog=prog),
        interval=log_interval,
        agent_names=[a[0] for a in DEMO_AGENTS],
    )
    print(f"opening Zellij demo ({layout})", file=sys.stderr)
    launch_zellij(layout, session=ZELLIJ_DEMO_SESSION)


async def _chatter_loop(
    cfg: WorkspaceConfig,
    *,
    interval: float,
    stop: asyncio.Event,
) -> None:
    i = 0
    while not stop.is_set():
        from_name, to_name, message = CHATTER_SCRIPT[i % len(CHATTER_SCRIPT)]
        i += 1
        try:
            proxy = await resolve_agent(
                pul.get_system(),
                to_name,
                workspace_id=cfg.cluster_id,
                timeout=30.0,
            )
            await proxy.deliver_message(
                from_name,
                message,
                channel="whisper",
                wait=False,
            )
        except Exception as e:
            print(f"chatter {from_name}→{to_name}: {e!r}", file=sys.stderr)
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(2.0, interval))
        except asyncio.TimeoutError:
            pass


async def run_demo(args: Namespace, *, prog: str = DEFAULT_PROG) -> None:
    root = Path.cwd().resolve()
    cfg = prepare_demo_workspace(root, prog=prog)
    try_exec_demo_zellij(cfg, args, prog=prog)

    llm = _demo_llm_options(cfg, args)
    interval = float(getattr(args, "interval", 6.0))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop.set())

    print(
        f"demo world {cfg.cluster_id} — agents: {', '.join(a[0] for a in DEMO_AGENTS)}",
        file=sys.stderr,
    )
    if llm["provider"] == "demo":
        print(
            "using demo LLM (no API key) — pass --real-llm for live models",
            file=sys.stderr,
        )

    try:
        async with workspace_session(
            cfg, bind_addr=getattr(args, "addr", "127.0.0.1:0")
        ) as system:
            write_node_record(cfg, addr=str(system.addr), pid=os.getpid())
            await spawn_shared_tool_worker(cfg)
            for name, npc_class, role in DEMO_AGENTS:
                await spawn_npc(
                    cfg,
                    name,
                    llm,
                    role=role,
                    npc_class=npc_class,
                    shared_tool_worker=True,
                )
            chatter = asyncio.create_task(
                _chatter_loop(cfg, interval=interval, stop=stop),
            )
            print(
                f"awake at {system.addr} — chattering every {interval}s (Ctrl+C to stop)",
                file=sys.stderr,
            )
            await stop.wait()
            chatter.cancel()
            try:
                await chatter
            except asyncio.CancelledError:
                pass
    finally:
        clear_node_record(cfg)
