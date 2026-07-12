# SPDX-License-Identifier: Apache-2.0
"""Lightweight per-agent activity snapshot for cluster watch."""

from __future__ import annotations

import time
from typing import Any

from pulsing.agent.actors.log import append_log

ActivityState = str  # idle | starting | thinking | tool | unknown


def _activity_line(state: str, detail: str, tool: str, from_sender: str) -> str:
    who = f" from {from_sender}" if from_sender else ""
    if state == "tool" and tool:
        body = detail or tool
        return f"⚙ {tool}: {body}"
    if state == "thinking":
        return f"… thinking{who}" + (f" — {detail}" if detail else "")
    if state == "idle":
        return "● idle"
    if state == "starting":
        return "○ starting"
    return f"{state}: {detail or tool or '—'}"


def init_activity(agent: Any) -> None:
    agent._activity = _snapshot("starting", detail="booting")


def set_activity(
    agent: Any,
    *,
    state: ActivityState,
    detail: str = "",
    from_sender: str = "",
    tool: str = "",
) -> None:
    prev = getattr(agent, "_activity", None) or {}
    now = time.time()
    same_state = prev.get("state") == state and prev.get("tool") == tool
    agent._activity = {
        "state": state,
        "detail": (detail or "").strip(),
        "from": (from_sender or prev.get("from") or "").strip(),
        "tool": (tool or "").strip(),
        "since": prev.get("since", now) if same_state else now,
        "updated_at": now,
    }
    if not same_state or (detail or "").strip() != (prev.get("detail") or "").strip():
        append_log(
            agent,
            _activity_line(
                state,
                agent._activity["detail"],
                agent._activity["tool"],
                agent._activity["from"],
            ),
        )


def get_activity(agent: Any) -> dict[str, Any]:
    snap = dict(getattr(agent, "_activity", _snapshot("unknown")))
    snap.setdefault("name", getattr(agent, "_cluster_short_name", "") or "")
    snap.setdefault("npc_class", getattr(agent, "_npc_class", "") or "")
    snap.setdefault("role", getattr(agent, "_agent_role", "") or "")
    return snap


def _snapshot(state: ActivityState, *, detail: str = "") -> dict[str, Any]:
    now = time.time()
    return {
        "state": state,
        "detail": detail,
        "from": "",
        "tool": "",
        "since": now,
        "updated_at": now,
    }
