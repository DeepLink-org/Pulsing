# SPDX-License-Identifier: Apache-2.0
"""Collect and format live activity across workspace NPCs."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pulsing.agent.cluster.discovery import list_cluster_agents
from pulsing.agent.cluster.resolve import resolve_agent

_STATE_LABEL = {
    "idle": "idle",
    "starting": "boot",
    "thinking": "think",
    "tool": "tool",
    "unknown": "?",
    "offline": "off",
}


def _ago(seconds: float) -> str:
    if seconds < 1:
        return "now"
    if seconds < 60:
        return f"{int(seconds)}s"
    return f"{int(seconds // 60)}m"


def format_activity_table(
    rows: list[dict[str, Any]], *, title: str = "Cluster activity"
) -> str:
    if not rows:
        return "(no agents — run `pulsing agent wake` or `pulsing agent spawn NAME`)"
    lines = [
        title + ":",
        f"{'NPC':<16} {'CLASS':<12} {'STATE':<6} {'FOR':<10} {'AGE':<5} DETAIL",
    ]
    for r in rows:
        state = str(r.get("state") or "unknown")
        label = _STATE_LABEL.get(state, state[:6])
        detail = str(r.get("detail") or r.get("error") or "")
        tool = str(r.get("tool") or "")
        if tool and state == "tool":
            detail = f"{tool}: {detail}" if detail else tool
        since = float(r.get("since") or r.get("updated_at") or time.time())
        age = _ago(max(0.0, time.time() - since))
        from_who = str(r.get("from") or "—")[:10]
        lines.append(
            f"{str(r.get('name') or '?'):<16} "
            f"{str(r.get('npc_class') or '-'):<12} "
            f"{label:<6} "
            f"{from_who:<10} "
            f"{age:<5} "
            f"{detail[:80]}"
        )
    return "\n".join(lines)


async def _fetch_one(
    system: Any,
    row: dict[str, Any],
    *,
    workspace_id: str,
    timeout: float,
) -> dict[str, Any]:
    name = str(row.get("name") or "")
    base = {
        "name": name,
        "node_id": row.get("node_id"),
        "instance_count": row.get("instance_count", 0),
    }
    if int(row.get("instance_count") or 0) <= 0:
        return {**base, "state": "offline", "detail": "no live instance"}
    try:
        proxy = await resolve_agent(
            system,
            name,
            workspace_id=workspace_id,
            timeout=min(timeout, 15.0),
        )
        act = await proxy.get_activity()
        if isinstance(act, dict):
            return {**base, **act}
        return {**base, "state": "unknown", "detail": str(act)}
    except Exception as e:
        return {**base, "state": "unknown", "detail": repr(e)}


async def collect_cluster_activity(
    system: Any,
    *,
    workspace_id: str,
    local_node_only: bool = False,
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    rows = await list_cluster_agents(
        system,
        workspace_id=workspace_id,
        local_node_only=local_node_only,
    )
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(row)
    if not unique:
        return []
    results = await asyncio.gather(
        *(
            _fetch_one(system, row, workspace_id=workspace_id, timeout=timeout)
            for row in unique
        ),
    )
    out = list(results)
    out.sort(key=lambda r: (r.get("state") != "idle", r.get("name") or ""))
    return out
