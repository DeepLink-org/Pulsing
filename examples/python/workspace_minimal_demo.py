#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI entry for :mod:`pulsing.workspace.minimal_demo`.

    uv run python examples/python/workspace_minimal_demo.py

Two-terminal flow (after ``pulsing init``):

    pulsing agent wake --provider demo --agents guide   # terminal 1
    pulsing agent say guide "list project files"        # terminal 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from pulsing.workspace.minimal_demo import run_workspace_minimal_demo


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=None, help="workspace directory")
    p.add_argument(
        "--message",
        default="list project files with Glob",
        help="message sent to guide",
    )
    p.add_argument(
        "--provider",
        default="demo",
        choices=("demo", "anthropic", "openai"),
    )
    p.add_argument("--model", default=None)
    p.add_argument("--template", default="agent", choices=("agent", "minimal"))
    p.add_argument("--keep", action="store_true")
    return p.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="pulsing-ws-demo-")
        root = Path(temp_dir.name)
    else:
        root = args.dir

    print(f"# workspace: {root}", file=sys.stderr)
    if args.provider == "demo":
        print("# LLM: demo (offline, no API key)", file=sys.stderr)
    else:
        print(f"# LLM: {args.provider}/{args.model or 'default'}", file=sys.stderr)

    try:
        out = await run_workspace_minimal_demo(
            root,
            message=args.message,
            provider=args.provider,
            model=args.model,
            template=args.template,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    body = out.get("assistant_text") or out.get("error") or out
    print(f"\nguide › {body}\n")
    if out.get("error"):
        return 1

    if temp_dir is not None and args.keep:
        temp_dir.cleanup = lambda: None  # type: ignore[method-assign]
        print(f"# kept workspace at {root}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
