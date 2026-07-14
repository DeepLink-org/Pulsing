# SPDX-License-Identifier: Apache-2.0
"""Quest (puzzle) state: assign_to, status, agent reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pulsing.agent.workspace.config import WorkspaceConfig, load_config, save_config

QUEST_STATUSES = frozenset({"open", "in_progress", "solved", "failed"})


def normalize_quest(puzzle: dict[str, Any]) -> dict[str, str]:
    out = {k: str(v) for k, v in puzzle.items()}
    status = (out.get("status") or "open").strip().lower()
    if status not in QUEST_STATUSES:
        status = "open"
    out["status"] = status
    out.setdefault("assign_to", "")
    return out


def quests_for_agent(
    cfg: WorkspaceConfig, agent_name: str
) -> list[tuple[str, dict[str, str]]]:
    short = (agent_name or "").strip()
    if not short:
        return []
    out: list[tuple[str, dict[str, str]]] = []
    for qid, raw in cfg.puzzles.items():
        p = normalize_quest(raw)
        assign = (p.get("assign_to") or "").strip()
        if assign and assign == short:
            out.append((qid, p))
    return sorted(out)


def format_quest_brief(qid: str, puzzle: dict[str, str]) -> str:
    title = puzzle.get("title") or qid
    status = puzzle.get("status") or "open"
    kind = puzzle.get("kind") or "task"
    path = puzzle.get("path") or "."
    line = f"{qid} [{kind}/{status}] {title} @ {path}"
    if puzzle.get("blurb"):
        line += f" — {puzzle['blurb']}"
    return line


def quest_context_for_agent(cfg: WorkspaceConfig, agent_name: str) -> str:
    assigned = quests_for_agent(cfg, agent_name)
    if not assigned:
        return ""
    lines = ["Assigned quests:"]
    for qid, p in assigned:
        lines.append(f"- {format_quest_brief(qid, p)}")
    lines.append("Use QuestReport to update status when progress changes.")
    return "\n".join(lines)


def update_quest_status(
    root: Path,
    quest_id: str,
    *,
    status: str,
    note: str = "",
    reporter: str = "",
) -> dict[str, str]:
    st = (status or "").strip().lower()
    if st not in QUEST_STATUSES:
        raise ValueError(f"invalid status {status!r}")
    cfg = load_config(root)
    puzzle = cfg.puzzles.get(quest_id)
    if puzzle is None:
        raise KeyError(f"unknown quest {quest_id!r}")
    updated = normalize_quest(puzzle)
    updated["status"] = st
    if note.strip():
        updated["last_note"] = note.strip()
    if reporter.strip():
        updated["last_reporter"] = reporter.strip()
    cfg.puzzles[quest_id] = updated
    save_config(cfg)
    return updated
