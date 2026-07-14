# SPDX-License-Identifier: Apache-2.0
"""Agent naming: workspace-scoped ``<prefix>/<workspace_id>/<name>``."""

from __future__ import annotations

from pulsing.forge.naming import DEFAULT_WORKSPACE_PREFIX

WS_AGENT_PREFIX = f"{DEFAULT_WORKSPACE_PREFIX.strip('/')}/"


def workspace_agent_name(workspace_id: str, short: str) -> str:
    ws = (workspace_id or "").strip().strip("/")
    s = (short or "").strip().strip("/")
    if not ws:
        raise ValueError("workspace_id must be non-empty")
    if not s:
        raise ValueError("agent name must be non-empty")
    if s.startswith(WS_AGENT_PREFIX):
        return s
    if s.startswith(f"{WS_AGENT_PREFIX}{ws}/"):
        return s
    return f"{WS_AGENT_PREFIX}{ws}/{s}"


def full_agent_name(short: str, *, workspace_id: str) -> str:
    return workspace_agent_name(workspace_id, short)


def is_public_npc_name(short: str) -> bool:
    """True for player-visible NPC ids (exclude ``_engine``, ``tasks/…`` internals)."""
    s = (short or "").strip()
    if not s or s.startswith("_") or "/" in s:
        return False
    return True


def workspace_list_prefix(workspace_id: str) -> str:
    return f"{WS_AGENT_PREFIX}{workspace_id.strip('/')}/"


def short_agent_name(full_or_short: str, *, workspace_id: str | None = None) -> str:
    s = (full_or_short or "").strip()
    if workspace_id:
        pref = workspace_list_prefix(workspace_id)
        if s.startswith(pref):
            return s[len(pref) :] or s
    if s.startswith(WS_AGENT_PREFIX):
        rest = s[len(WS_AGENT_PREFIX) :]
        if "/" in rest:
            return rest.split("/", 1)[1]
        return rest or s
    return s
