# SPDX-License-Identifier: Apache-2.0
"""Workspace = world: ``.pulsing/cluster.json`` holds cluster + puzzles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsing.agent.workspace.root import (
    CLUSTER_FILE,
    NODE_FILE,
    PULSING_DIR,
    workspace_cluster_id,
)

DEFAULT_PUZZLES: dict[str, dict[str, str]] = {
    "unit-tests": {
        "title": "Unit test suite",
        "kind": "test",
        "path": "tests",
        "blurb": "Run pytest; keep green.",
        "status": "open",
        "assign_to": "",
    },
}


@dataclass
class WorkspaceConfig:
    root: str
    cluster_id: str
    name: str = ""
    provider: str = "anthropic"
    model: str | None = None
    auto_approve: bool = False
    sandbox: str = "off"
    default_agents: list[str] = field(default_factory=lambda: ["guide"])
    shared_tool_worker: bool = False
    puzzles: dict[str, dict[str, str]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_PUZZLES.items()},
    )

    @property
    def pulsing_dir(self) -> Path:
        return Path(self.root) / PULSING_DIR

    @property
    def cluster_path(self) -> Path:
        return self.pulsing_dir / CLUSTER_FILE

    @property
    def node_path(self) -> Path:
        return self.pulsing_dir / NODE_FILE

    def seed_addr(self) -> str | None:
        p = self.node_path
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        addr = str(data.get("addr") or "").strip()
        return addr or None


def default_config(root: Path) -> WorkspaceConfig:
    root = root.resolve()
    return WorkspaceConfig(
        root=str(root),
        cluster_id=workspace_cluster_id(root),
        name=root.name or "workspace",
    )


def load_config(root: Path) -> WorkspaceConfig:
    path = root / PULSING_DIR / CLUSTER_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    puzzles = data.get("puzzles") or DEFAULT_PUZZLES
    return WorkspaceConfig(
        root=str(root.resolve()),
        cluster_id=str(data.get("cluster_id") or workspace_cluster_id(root)),
        name=str(data.get("name") or root.name or ""),
        provider=str(data.get("provider") or "anthropic"),
        model=data.get("model"),
        auto_approve=bool(data.get("auto_approve", False)),
        sandbox=str(data.get("sandbox") or "off"),
        default_agents=list(data.get("default_agents") or ["guide"]),
        shared_tool_worker=bool(data.get("shared_tool_worker", False)),
        puzzles={k: dict(v) for k, v in puzzles.items()},
    )


def save_config(cfg: WorkspaceConfig) -> None:
    cfg.pulsing_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "cluster_id": cfg.cluster_id,
        "name": cfg.name,
        "provider": cfg.provider,
        "model": cfg.model,
        "auto_approve": cfg.auto_approve,
        "sandbox": cfg.sandbox,
        "default_agents": cfg.default_agents,
        "shared_tool_worker": cfg.shared_tool_worker,
        "puzzles": cfg.puzzles,
    }
    cfg.cluster_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_node_record(cfg: WorkspaceConfig, *, addr: str, pid: int) -> None:
    cfg.pulsing_dir.mkdir(parents=True, exist_ok=True)
    cfg.node_path.write_text(
        json.dumps({"addr": addr, "pid": pid}, indent=2) + "\n",
        encoding="utf-8",
    )


def clear_node_record(cfg: WorkspaceConfig) -> None:
    try:
        cfg.node_path.unlink(missing_ok=True)
    except OSError:
        pass
