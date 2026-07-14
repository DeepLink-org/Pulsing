# SPDX-License-Identifier: Apache-2.0
"""Forge event inbox — decouple tell delivery from host mailbox."""

from __future__ import annotations

import logging
from typing import Any

from pulsing.core import ActorProxy, remote, resolve

from pulsing.forge.events import ForgeEvent, ForgeEventKind
from pulsing.forge.naming import forge_event_inbox_name

logger = logging.getLogger(__name__)

_STREAM_KINDS = frozenset(
    {
        ForgeEventKind.EXEC_OUTPUT_DELTA.value,
        ForgeEventKind.TOOL_BEGIN.value,
        ForgeEventKind.TOOL_END.value,
        ForgeEventKind.USER_INPUT_REQUEST.value,
    }
)


@remote
class ForgeEventInbox:
    """Collects Forge events and forwards side effects to the host agent."""

    def __init__(self, host_name: str) -> None:
        self._host_name = (host_name or "").strip()
        self._events: list[dict[str, Any]] = []

    async def on_forge_event(self, event: dict[str, Any]) -> None:
        raw = dict(event)
        self._events.append(raw)
        if not self._host_name:
            return
        kind = str(raw.get("kind") or "")
        try:
            host = await resolve(self._host_name)
            if kind in _STREAM_KINDS:
                await host.as_any().tell("on_forge_stream_event", raw)
            else:
                await host.as_any().tell("on_forge_side_effect", raw)
        except Exception as exc:
            logger.warning(
                "forge inbox forward failed host=%s kind=%s: %s",
                self._host_name,
                kind,
                exc,
            )

    def get_forge_events(self, since: int = 0) -> list[dict[str, Any]]:
        if since <= 0:
            return list(self._events)
        return self._events[since:]

    def event_count(self) -> int:
        return len(self._events)


async def ensure_forge_event_inbox(host_name: str) -> ActorProxy:
    """Resolve or spawn ``{host}/events``."""
    name = forge_event_inbox_name(host_name)
    try:
        return await resolve(name, cls=ForgeEventInbox, timeout=30.0)
    except Exception:
        return await ForgeEventInbox.spawn(host_name, name=name, public=True)
