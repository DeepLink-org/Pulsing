#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal Pulsing Forge demo (``pulsing.forge``; no Craft)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pulsing as pul
from pulsing.forge import ForgeEnvironment, ToolWorkerActor, ToolWorkerConfig


async def main() -> None:
    root = Path.cwd()
    env = ForgeEnvironment(cwd=str(root), sandbox_policy="off")
    rt = env.runtime()

    print("== local environment ==")
    print(rt.call_tool("Glob", {"pattern": "README*", "path": str(root)}).content[:200])

    print("== actor worker ==")
    await pul.init()
    try:
        worker = await ToolWorkerActor.spawn(
            ToolWorkerConfig(cwd=str(root)),
            public=False,
        )
        out = await worker.Glob(pattern="pyproject.toml", path=str(root))
        print(out.get("content", out))
    finally:
        await pul.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
