# SPDX-License-Identifier: Apache-2.0
"""Deliver Forge events to a named Pulsing actor (point-to-point tell)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pulsing.core.remote import resolve

from pulsing._async_bridge import run_sync
from pulsing.forge.events import ForgeEvent

logger = logging.getLogger(__name__)


async def tell_forge_event(sink_name: str, event: ForgeEvent | dict[str, Any]) -> None:
    payload = event.to_dict() if isinstance(event, ForgeEvent) else dict(event)
    proxy = await resolve(sink_name)
    await proxy.as_any().tell("on_forge_event", payload)


async def ask_exec_approval(sink_name: str, request: dict[str, Any]) -> dict[str, Any]:
    """Blocking approval RPC: isolated worker → host Agent."""
    if not sink_name:
        return {"decision": "denied"}
    proxy = await resolve(sink_name)
    out = await proxy.as_any().ask("resolve_exec_approval", dict(request))
    return dict(out) if isinstance(out, dict) else {"decision": "denied"}


async def ask_request_permissions(
    sink_name: str, args: dict[str, Any]
) -> dict[str, Any]:
    if not sink_name:
        raise RuntimeError("request_permissions requires approval sink")
    proxy = await resolve(sink_name)
    out = await proxy.as_any().ask("resolve_request_permissions", dict(args))
    if not isinstance(out, dict):
        raise RuntimeError("invalid request_permissions response")
    return dict(out)


def tell_forge_event_sync(sink_name: str, event: ForgeEvent | dict[str, Any]) -> None:
    """Best-effort tell from sync code (e.g. exec reader threads)."""
    if not sink_name:
        return
    try:
        run_sync(tell_forge_event(sink_name, event), timeout=5.0)
    except Exception as e:
        logger.debug("forge p2p tell failed sink=%s: %s", sink_name, e)


def ask_exec_approval_sync(sink_name: str, request: dict[str, Any]) -> dict[str, Any]:
    if not sink_name:
        return {"decision": "denied"}
    try:
        return run_sync(ask_exec_approval(sink_name, request), timeout=120.0)
    except Exception as e:
        logger.warning("forge exec approval ask failed sink=%s: %s", sink_name, e)
        return {"decision": "denied"}


def ask_request_permissions_sync(
    sink_name: str, args: dict[str, Any]
) -> dict[str, Any]:
    if not sink_name:
        raise RuntimeError("request_permissions requires approval sink")
    try:
        return run_sync(ask_request_permissions(sink_name, args), timeout=120.0)
    except Exception as e:
        logger.warning("forge request_permissions ask failed sink=%s: %s", sink_name, e)
        raise


class ForgeEventPump:
    """Async queue + background task: batches tell to host without blocking tool threads."""

    def __init__(self, sink_name: str | None) -> None:
        self._sink = (sink_name or "").strip() or None
        self._queue: asyncio.Queue[ForgeEvent | None] | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._sink)

    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._queue is None:
            return
        await self._queue.put(None)
        if self._task is not None:
            await self._task
        self._task = None
        self._queue = None

    def emit_sync(self, event: ForgeEvent) -> None:
        if not self.enabled or self._queue is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:
            tell_forge_event_sync(self._sink or "", event)

    def emit(self, event: ForgeEvent) -> None:
        if not self.enabled or self._queue is None:
            return
        self._queue.put_nowait(event)

    async def _run(self) -> None:
        assert self._queue is not None
        assert self._sink is not None
        while True:
            event = await self._queue.get()
            if event is None:
                return
            try:
                await tell_forge_event(self._sink, event)
            except Exception as e:
                logger.warning("forge event pump tell failed: %s", e)
