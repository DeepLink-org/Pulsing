#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quickstart: ForgeAgent with zero API keys (demo provider).

    uv run python examples/python/forge_agent_quickstart.py

With OpenAI:

    OPENAI_API_KEY=sk-... uv run python examples/python/forge_agent_quickstart.py \\
        --provider openai --model gpt-4o-mini \\
        "Find README files and summarize the project in one paragraph"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pulsing.forge.host import ForgeAgent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt", nargs="?", default="List README files in this project.")
    p.add_argument("--cwd", type=Path, default=Path.cwd())
    p.add_argument(
        "--provider", default="demo", choices=["demo", "openai", "anthropic"]
    )
    p.add_argument("--model", default=None)
    p.add_argument("--quiet", action="store_true", help="Disable assistant streaming")
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()
    model = args.model
    if model is None:
        model = {
            "demo": "demo",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
        }[args.provider]

    from pulsing.forge.host.cli_events import CliEventSink

    events = CliEventSink(stream_assistant=not args.quiet)
    agent = ForgeAgent(
        cwd=args.cwd.resolve(),
        provider=args.provider,
        model=model,
        events=events,
    )
    try:
        print(f"# ForgeAgent ({args.provider}/{model})\n")
        answer = await agent.run(args.prompt)
        print(f"\n# done\n{answer}")
    finally:
        agent.close()


if __name__ == "__main__":
    asyncio.run(_main())
