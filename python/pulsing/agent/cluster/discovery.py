# SPDX-License-Identifier: Apache-2.0
"""Discover cc cluster agents via Pulsing gossip (``all_named_actors``)."""

from __future__ import annotations

from typing import Any

from pulsing.agent.cluster.constants import (
    WS_AGENT_PREFIX,
    is_public_npc_name,
    workspace_list_prefix,
)


def _path_to_name(path_str: str) -> str:
    if path_str.startswith("actors/"):
        return path_str[7:]
    return path_str


def _parse_ws_agent(name: str) -> tuple[str, str] | None:
    if not name.startswith(WS_AGENT_PREFIX):
        return None
    rest = name[len(WS_AGENT_PREFIX) :]
    if "/" not in rest:
        return None
    ws_id, short = rest.split("/", 1)
    if not ws_id or not short:
        return None
    return ws_id, short


async def list_cluster_agents(
    system: Any,
    *,
    workspace_id: str | None = None,
    local_node_only: bool = False,
) -> list[dict[str, Any]]:
    """Return agent rows; filter to one workspace when ``workspace_id`` is set."""
    all_named = await system.all_named_actors()
    local_nid = str(system.node_id.id)
    ws_prefix = workspace_list_prefix(workspace_id) if workspace_id else None
    out: list[dict[str, Any]] = []

    for info in all_named:
        path_str = str(info.get("path", ""))
        name = _path_to_name(path_str)
        parsed = _parse_ws_agent(name)
        if parsed is None:
            continue
        ws_id, short = parsed
        if ws_prefix and not name.startswith(ws_prefix):
            continue
        if workspace_id and ws_id != workspace_id.strip("/"):
            continue
        if not is_public_npc_name(short):
            continue

        instance_count = int(info.get("instance_count", 0) or 0)
        if instance_count <= 0:
            out.append(
                {
                    "name": short,
                    "full_name": name,
                    "instance_count": 0,
                    "node_id": None,
                    "actor_id": None,
                }
            )
            continue
        try:
            instances = await system.get_named_instances(name)
        except Exception:
            instances = []
        for inst in instances:
            nid = str(inst.get("node_id", ""))
            if local_node_only and nid != local_nid:
                continue
            out.append(
                {
                    "name": short,
                    "full_name": name,
                    "instance_count": instance_count,
                    "node_id": nid,
                    "actor_id": str(inst.get("actor_id", "")),
                }
            )
    out.sort(key=lambda r: (r.get("name") or "", r.get("node_id") or ""))
    return out


def format_agent_table(
    rows: list[dict[str, Any]],
    *,
    workspace_id: str | None = None,
) -> str:
    if not rows:
        if workspace_id:
            return "(no agents in this workspace; try `pulsing agent spawn NAME` or `pulsing agent wake`)"
        return "(no cluster agents registered)"
    lines = [f"{'NAME':<20} {'NODE':<12} {'ACTOR_ID':<36} INST"]
    for r in rows:
        lines.append(
            f"{r.get('name', ''):<20} {str(r.get('node_id') or '-'):<12} "
            f"{str(r.get('actor_id') or '-'):<36} {r.get('instance_count', 0)}"
        )
    return "\n".join(lines)
