# SPDX-License-Identifier: Apache-2.0
"""Agent spawn configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsing.agent.cluster.constants import full_agent_name, short_agent_name
from pulsing.agent.npc.loader import (
    NpcClass,
    load_npc_class,
    list_npc_classes as _list_npc_classes,
)
from pulsing.agent.workspace.quest import quest_context_for_agent
from pulsing.agent.workspace.config import load_config


def build_npc_prompt(
    *,
    short_name: str,
    workspace_id: str,
    cls: NpcClass,
    role: str = "",
    personality: str = "",
    description: str = "",
    cwd: str,
    quest_context: str = "",
) -> str:
    lines = [
        f"You are {short_name} in world {workspace_id}.",
        f"Class: {cls.name} — {cls.description}",
    ]
    if role:
        lines.append(f"Role: {role}.")
    if personality:
        lines.append(f"Personality: {personality}.")
    if description:
        lines.append(description)
    if cls.prompt_extra:
        lines.append(cls.prompt_extra)
    if quest_context:
        lines.append(quest_context)
    lines.append(f"Working directory: {cwd}")
    return "\n".join(lines)


def get_npc_class(name: str, workspace_root: Path | str | None = None) -> NpcClass:
    root = Path(workspace_root) if workspace_root else None
    return load_npc_class(name, root)


def list_npc_classes(workspace_root: Path | str | None = None) -> list[str]:
    root = Path(workspace_root) if workspace_root else None
    return _list_npc_classes(root)


@dataclass
class NpcConfig:
    model: str
    cwd: str
    agent_name: str
    workspace_id: str
    provider: str = "anthropic"
    api_key: str | None = None
    base_url: str | None = None
    auto_approve: bool = False
    system_prompt: str | None = None
    prompt_callback: Any | None = None
    sandbox_policy: str = "off"
    dangerously_disable_sandbox: bool = False
    agent_role: str = ""
    agent_description: str = ""
    summon_depth: int = 0
    max_summon_depth: int = 3
    tool_allowlist: list[str] | None = None
    tool_forbid: list[str] = field(default_factory=list)
    npc_class: str = "artisan"
    personality: str = ""
    shared_tool_worker: bool = False

    @property
    def short_name(self) -> str:
        return short_agent_name(self.agent_name, workspace_id=self.workspace_id)

    @property
    def workspace_root(self) -> Path:
        return Path(self.cwd)

    def full_name(self) -> str:
        return full_agent_name(self.short_name, workspace_id=self.workspace_id.strip())

    def resolved_class(self) -> NpcClass:
        return load_npc_class(self.npc_class or "artisan", self.workspace_root)

    def resolved_profile(self) -> tuple[str, list[str], set[str], str, str]:
        """Return (system_prompt, tool_allowlist, tool_forbid, npc_class_name, personality)."""
        cls_def = self.resolved_class()
        npc_class_name = cls_def.name
        personality = (self.personality or cls_def.default_personality).strip()
        ws = self.workspace_id.strip()
        short = self.short_name
        quest_ctx = ""
        cluster_path = self.workspace_root / ".pulsing" / "cluster.json"
        if cluster_path.is_file():
            try:
                quest_ctx = quest_context_for_agent(
                    load_config(self.workspace_root), short
                )
            except OSError:
                pass
        system_prompt = (self.system_prompt or "").strip()
        if not system_prompt:
            system_prompt = build_npc_prompt(
                short_name=short,
                workspace_id=ws,
                cls=cls_def,
                role=self.agent_role.strip(),
                personality=personality,
                description=self.agent_description.strip(),
                cwd=self.cwd,
                quest_context=quest_ctx,
            )
        allow = (
            list(self.tool_allowlist)
            if self.tool_allowlist is not None
            else list(cls_def.default_tools)
        )
        forbid = set(self.tool_forbid) | set(cls_def.forbidden_tools)
        return system_prompt, allow, forbid, npc_class_name, personality


AgentConfig = NpcConfig

__all__ = [
    "AgentConfig",
    "NpcClass",
    "NpcConfig",
    "build_npc_prompt",
    "get_npc_class",
    "list_npc_classes",
]
