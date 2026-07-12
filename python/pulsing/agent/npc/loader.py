# SPDX-License-Identifier: Apache-2.0
"""Load NpcClass from ``.pulsing/npcs/*.json`` with built-in defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsing.agent.workspace.root import PULSING_DIR

NPCS_SUBDIR = "npcs"

_READ = ["Read", "Glob", "Grep"]
_WRITE = ["Edit", "Write", "Bash"]
_PEER = ["Summon", "MessageClusterAgent", "ListClusterAgents", "QuestReport"]

_BUILTIN: dict[str, dict[str, Any]] = {
    "artisan": {
        "name": "artisan",
        "description": "工匠 — 读写文件、执行命令。",
        "default_personality": "helpful and precise",
        "prompt_extra": "",
        "default_tools": _READ + _WRITE,
        "forbidden_tools": [],
        "model_hint": None,
    },
    "quest_giver": {
        "name": "quest_giver",
        "description": "任务发布者 — 召唤与协调其他 NPC。",
        "default_personality": "organized and strategic",
        "prompt_extra": "Use Summon to spawn helpers, MessageClusterAgent to coordinate peers, QuestReport to track quests.",
        "default_tools": _PEER + _READ,
        "forbidden_tools": [],
        "model_hint": None,
    },
    "scholar": {
        "name": "scholar",
        "description": "学者 — 只读审查。",
        "default_personality": "critical and detail-oriented",
        "prompt_extra": "",
        "default_tools": _READ + ["FetchUrl", "QuestReport"],
        "forbidden_tools": _WRITE,
        "model_hint": None,
    },
    "oracle": {
        "name": "oracle",
        "description": "先知 — 信息收集。",
        "default_personality": "curious and resourceful",
        "prompt_extra": "",
        "default_tools": _READ + ["FetchUrl"],
        "forbidden_tools": _WRITE + ["Summon"],
        "model_hint": None,
    },
}


@dataclass
class NpcClass:
    name: str
    description: str
    default_personality: str = ""
    prompt_extra: str = ""
    default_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    model_hint: str | None = None


def npcs_dir(workspace_root: Path | None) -> Path | None:
    if workspace_root is None:
        return None
    return workspace_root / PULSING_DIR / NPCS_SUBDIR


def _from_dict(data: dict[str, Any]) -> NpcClass:
    name = str(data.get("name") or "artisan").strip().lower()
    return NpcClass(
        name=name,
        description=str(data.get("description") or ""),
        default_personality=str(data.get("default_personality") or ""),
        prompt_extra=str(data.get("prompt_extra") or ""),
        default_tools=[str(t) for t in (data.get("default_tools") or [])],
        forbidden_tools=[str(t) for t in (data.get("forbidden_tools") or [])],
        model_hint=(
            (str(data["model_hint"]).strip() or None)
            if data.get("model_hint")
            else None
        ),
    )


def load_npc_class(name: str, workspace_root: Path | None = None) -> NpcClass:
    key = (name or "artisan").strip().lower()
    nd = npcs_dir(workspace_root)
    if nd is not None:
        path = nd / f"{key}.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return _from_dict(data)
            except (OSError, json.JSONDecodeError):
                pass
    raw = _BUILTIN.get(key) or _BUILTIN["artisan"]
    return _from_dict(raw)


def list_npc_classes(workspace_root: Path | None = None) -> list[str]:
    names: set[str] = set(_BUILTIN.keys())
    nd = npcs_dir(workspace_root)
    if nd is not None and nd.is_dir():
        for path in nd.glob("*.json"):
            names.add(path.stem.lower())
    return sorted(names)


def seed_npc_defs(workspace_root: Path) -> None:
    """Write bundled npc json files when missing."""
    nd = npcs_dir(workspace_root)
    assert nd is not None
    nd.mkdir(parents=True, exist_ok=True)
    for key, data in _BUILTIN.items():
        path = nd / f"{key}.json"
        if path.is_file():
            continue
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
