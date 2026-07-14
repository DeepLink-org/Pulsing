# SPDX-License-Identifier: Apache-2.0
"""Forge approval helpers for worker ↔ host RPC."""

from __future__ import annotations

from typing import Any

from pulsing.forge.events import ForgeEvent
from pulsing.forge.permissions import is_permission_profile_effectively_empty
from pulsing.forge.p2p_transport import (
    ask_exec_approval_sync,
    ask_request_permissions_sync,
    tell_forge_event_sync,
)


def _parse_exec_decision(raw: dict[str, Any]) -> str:
    return str(raw.get("decision") or "denied")


def make_worker_exec_approval_callback(sink_name: str | None):
    def _cb(request: dict[str, Any]) -> str:
        if sink_name:
            tell_forge_event_sync(sink_name, ForgeEvent.exec_approval_request(request))
            return _parse_exec_decision(ask_exec_approval_sync(sink_name, request))
        return "denied"

    return _cb


def make_worker_permissions_callback(sink_name: str | None):
    def _cb(args: dict[str, Any]) -> dict[str, Any]:
        if not sink_name:
            raise RuntimeError("request_permissions requires approval sink")
        tell_forge_event_sync(sink_name, ForgeEvent.request_permissions(args))
        out = ask_request_permissions_sync(sink_name, args)
        perms = out.get("permissions") or {}
        if is_permission_profile_effectively_empty(perms):
            raise RuntimeError("permissions denied by host")
        return out

    return _cb
