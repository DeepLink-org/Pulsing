# SPDX-License-Identifier: Apache-2.0
"""Player / look / puzzles — thin view over :class:`WorkspaceConfig`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pulsing.agent.workspace.config import WorkspaceConfig
from pulsing.agent.workspace.quest import normalize_quest


def player_name() -> str:
    return (
        os.environ.get("PLAYER")
        or os.environ.get("HERO")
        or os.environ.get("USER")
        or "player"
    ).strip() or "player"


def rel_path(root: Path, cwd: Path | None = None) -> str:
    base = root.resolve()
    here = (cwd or Path.cwd()).resolve()
    try:
        r = here.relative_to(base)
    except ValueError:
        return "."
    return "." if not r.parts else r.as_posix()


def puzzles_at(
    cfg: WorkspaceConfig, cwd: Path | None = None
) -> list[tuple[str, dict[str, str]]]:
    """Puzzles whose ``path`` matches current directory."""
    here = rel_path(Path(cfg.root), cwd).strip("./")
    out: list[tuple[str, dict[str, str]]] = []
    for pid, raw in cfg.puzzles.items():
        p = normalize_quest(raw)
        pp = (p.get("path") or ".").strip().strip("/")
        if (
            not pp
            or pp == "."
            or here == pp
            or here.startswith(pp + "/")
            or pp.startswith(here + "/")
        ):
            out.append((pid, p))
    return out


def _quest_line(pid: str, p: dict[str, str]) -> str:
    kind = p.get("kind") or "task"
    title = p.get("title") or pid
    path = p.get("path") or "."
    status = p.get("status") or "open"
    assign = p.get("assign_to") or ""
    line = f"  {pid} [{kind}/{status}] {title} @ {path}"
    if assign:
        line += f" → {assign}"
    return line


def format_puzzles(
    cfg: WorkspaceConfig, *, cwd: Path | None = None, all_: bool = False
) -> str:
    items = sorted(cfg.puzzles.items()) if all_ else puzzles_at(cfg, cwd)
    if not items:
        return "(no puzzles — add to .pulsing/cluster.json)"
    lines = [f"Puzzles ({cfg.name}):"]
    for pid, raw in items:
        p = normalize_quest(raw)
        lines.append(_quest_line(pid, p))
        if p.get("blurb"):
            lines.append(f"    {p['blurb']}")
    return "\n".join(lines)


def render_look(
    cfg: WorkspaceConfig,
    *,
    cwd: Path | None = None,
    npc_rows: list[dict[str, Any]] | None = None,
) -> str:
    root = Path(cfg.root)
    here = rel_path(root, cwd)
    player = player_name()
    seed = cfg.seed_addr()
    lines = [
        f"═══ {cfg.name} ═══",
        f"player: {player}  ·  path: {here}  ·  {root if here == '.' else root / here}",
    ]
    if cfg.shared_tool_worker:
        lines.append("tools: shared isolated worker")
    lines.append(
        f"node: {seed}" if seed else "node: (sleeping — run `pulsing agent wake`)",
    )
    local = puzzles_at(cfg, cwd)
    if local:
        lines.append("\nQuests here:")
        for pid, raw in local:
            lines.append(_quest_line(pid, normalize_quest(raw)))
    if npc_rows:
        lines.append("\nNPCs:")
        for r in npc_rows:
            lines.append(f"  {r.get('name', '?')}")
    elif seed:
        lines.append(
            "\nAgents: (none — `pulsing agent wake` or `pulsing agent spawn NAME`)"
        )
    lines.append("\nCommands: pulsing agent dashboard · watch · say …")
    return "\n".join(lines)
