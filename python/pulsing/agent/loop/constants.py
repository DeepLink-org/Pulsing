# SPDX-License-Identifier: Apache-2.0
"""Shared tool name sets for craft runtime routing."""

from __future__ import annotations

from pulsing.forge.integrated import (
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
    FORGE_TOOL_NAMES,
)

# Back-compat alias: all Forge tools that run in ToolWorkerActor.
ISOLATED_TOOL_NAMES: frozenset[str] = FORGE_ISOLATED_TOOL_NAMES

CLUSTER_TOOL_NAMES: frozenset[str] = frozenset(
    {"ListClusterAgents", "MessageClusterAgent"},
)

NPC_TOOL_NAMES: frozenset[str] = frozenset({"Summon"})

QUEST_TOOL_NAMES: frozenset[str] = frozenset({"QuestReport"})
