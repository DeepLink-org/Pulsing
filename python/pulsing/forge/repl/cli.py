# SPDX-License-Identifier: Apache-2.0
"""CLI: ``pulsing forge repl`` / ``python -m pulsing.forge.repl``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pulsing.forge.repl.rust_dispatch import try_run_rust_repl
from pulsing.forge.repl.session import ForgeReplSession
from pulsing.forge.repl.shell import run_repl

_PROG = "pulsing forge repl"


def build_parser(prog: str = _PROG) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Forge session REPL — manual tool calls and trace replay.",
    )
    p.add_argument("--cwd", default=".", help="workspace root (default: .)")
    p.add_argument(
        "--sandbox",
        default="off",
        choices=("off", "restricted", "bwrap"),
        help="sandbox policy",
    )
    p.add_argument(
        "--dangerously-disable-sandbox",
        action="store_true",
        help="disable sandbox even when policy is set",
    )
    p.add_argument(
        "--approve",
        choices=("auto", "ask"),
        default="auto",
        help="exec / user_input / plugin approval (default: auto)",
    )
    p.add_argument("--trace", help="JSONL trace to load for replay")
    p.add_argument(
        "--record",
        help="append tool calls / events to JSONL while in REPL",
    )
    p.add_argument(
        "--fork",
        type=int,
        metavar="N",
        help="after loading --trace, replay tool calls 1..N then enter interactive",
    )
    p.add_argument(
        "--replay-all",
        action="store_true",
        help="with --trace: run all tool calls then exit (no interactive)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="with --replay-all: print calls only",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="with --replay-all: compare results to trace",
    )
    p.add_argument(
        "--python",
        action="store_true",
        help="force Python REPL (skip Rust binary even if installed)",
    )
    return p


def _argv_for_rust(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.cwd != ".":
        out.extend(["--cwd", args.cwd])
    if args.sandbox != "off":
        out.extend(["--sandbox", args.sandbox])
    if args.dangerously_disable_sandbox:
        out.append("--dangerously-disable-sandbox")
    if args.approve != "auto":
        out.extend(["--approve", args.approve])
    if args.trace:
        out.extend(["--trace", args.trace])
    if args.record:
        out.extend(["--record", args.record])
    if args.replay_all:
        out.append("--replay-all")
    if args.dry_run:
        out.append("--dry-run")
    if args.verify:
        out.append("--verify")
    return out


def main(argv: list[str] | None = None) -> None:
    raw = list(argv if argv is not None else sys.argv[1:])
    args = build_parser().parse_args(raw)

    if not args.python and args.fork is None:
        rust_argv = _argv_for_rust(args)
        code = try_run_rust_repl(rust_argv)
        if code is not None:
            raise SystemExit(code)

    session = ForgeReplSession(
        cwd=Path(args.cwd),
        sandbox_policy=args.sandbox,
        dangerously_disable_sandbox=args.dangerously_disable_sandbox,
        approval_mode=args.approve,
        record_path=Path(args.record) if args.record else None,
    )
    if args.trace:
        session.load_replay_trace(args.trace)
    if args.fork is not None and args.trace:
        session.fork_trace(args.fork)
    if args.replay_all and args.trace:
        for line in session.replay_all(dry_run=args.dry_run, verify=args.verify):
            print(line)
        return
    run_repl(session)


if __name__ == "__main__":
    main()
