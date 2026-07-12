# SPDX-License-Identifier: Apache-2.0
"""LLM-guided workspace bootstrap after ``pulsing init``."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pulsing.forge.host.llm import default_model, default_provider

INIT_TOOL_NAMES: tuple[str, ...] = (
    "update_plan",
    "Glob",
    "Read",
    "Grep",
    "Write",
    "Edit",
    "shell_command",
)

INIT_SYSTEM = """You are bootstrapping a new Pulsing AI workspace.
The `.pulsing/` scaffold (cluster.json, hooks, journal) already exists.

Customize the project to match the user's goal:
1. Read `.pulsing/cluster.json` — adjust default_agents and puzzles if needed
2. Create or update project files (README.md, tests/, configs) with Write/Edit
3. Use Glob/Read to inspect before changing; keep changes minimal and practical
4. Do not delete `.pulsing/history/`
5. End with a short summary of what you configured
"""


def _resolve_provider_model(
    provider: str | None,
    model: str | None,
) -> tuple[str, str]:
    p = (provider or default_provider()).strip().lower()
    m = model or default_model(p)
    return p, m


async def run_init_guide(
    root: Path,
    guide: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    from pulsing.forge.host import ForgeAgent

    p, m = _resolve_provider_model(provider, model)
    agent = ForgeAgent(
        cwd=root,
        provider=p,
        model=m,
        tool_names=INIT_TOOL_NAMES,
        system_prompt=INIT_SYSTEM,
        auto_approve=True,
    )
    try:
        prompt = (
            f"Workspace bootstrap goal:\n\n{guide}\n\n"
            "Start by reading `.pulsing/cluster.json` and the project root, "
            "then apply changes."
        )
        return await agent.run(prompt)
    finally:
        agent.close()


def run_init_guide_sync(
    root: Path,
    guide: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_init_guide(root, guide, provider=provider, model=model))

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            run_init_guide(root, guide, provider=provider, model=model),
        ).result()
