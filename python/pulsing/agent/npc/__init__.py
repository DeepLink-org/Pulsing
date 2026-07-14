# SPDX-License-Identifier: Apache-2.0
"""Workspace NPC spawn (lazy-imports actor)."""

from __future__ import annotations

from typing import Any

from pulsing.agent.npc.config import NpcConfig, get_npc_class, list_npc_classes
from pulsing.agent.npc.loader import NpcClass, seed_npc_defs

__all__ = [
    "NpcClass",
    "NpcConfig",
    "spawn_npc",
    "get_npc_class",
    "list_npc_classes",
    "seed_npc_defs",
]


async def spawn_npc(
    config: NpcConfig,
    *,
    name: str | None = None,
    public: bool = True,
) -> Any:
    from pulsing.agent.actors import Agent

    return await Agent.spawn(
        config=config,
        name=name or config.full_name(),
        public=public,
    )
