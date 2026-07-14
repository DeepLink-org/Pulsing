# SPDX-License-Identifier: Apache-2.0
"""Agent spawn configuration."""

from __future__ import annotations

from typing import Any

from pulsing.agent.npc.config import (
    AgentConfig,
    NpcConfig,
    build_npc_prompt,
    get_npc_class,
    list_npc_classes,
)
from pulsing.agent.npc.loader import NpcClass

# Public alias; NpcConfig remains for backward compatibility.
__all__ = [
    "AgentConfig",
    "NpcClass",
    "NpcConfig",
    "build_npc_prompt",
    "get_npc_class",
    "list_npc_classes",
    "spawn_agent",
]


async def spawn_agent(
    config: AgentConfig,
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
