# SPDX-License-Identifier: Apache-2.0
"""Supervise isolated ToolWorkerActor lifecycle (respawn on failure)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pulsing as pul

from pulsing.forge.config import ToolWorkerConfig
from pulsing.forge.p2p_transport import ForgeEventPump
from pulsing.forge.worker import ToolWorkerActor

logger = logging.getLogger(__name__)


class ForgeWorkerSupervisor:
    """In-process supervisor for one ``ToolWorkerActor`` child process.

    Must run in the host process — ``new_process`` spawn cannot run safely
    from inside another actor's mailbox handler.
    """

    def __init__(self, worker_cfg: ToolWorkerConfig) -> None:
        self._cfg = worker_cfg
        self._pump = ForgeEventPump(worker_cfg.event_sink_name)
        self._spawn: pul.IsolatedSpawnHandle | None = None
        self._proxy: pul.ActorProxy | None = None
        self._lock = asyncio.Lock()

    @property
    def proxy(self) -> pul.ActorProxy | None:
        return self._proxy

    async def close(self) -> None:
        async with self._lock:
            await self._pump.stop()
            await self._teardown_locked()

    def terminate_process(self) -> None:
        if self._spawn is None:
            return
        proc = self._spawn.process
        if proc.returncode is None:
            proc.terminate()

    async def ping(self) -> dict[str, Any]:
        await self.ensure_ready(reason="ping")
        assert self._proxy is not None
        return await self._proxy.ping()

    async def ensure_ready(self, *, reason: str = "startup") -> None:
        async with self._lock:
            await self._ensure_worker_locked(reason=reason)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        event_sink: str | None = None,
    ) -> dict[str, Any]:
        last_exc: BaseException | None = None
        for attempt in range(2):
            try:
                async with self._lock:
                    await self._ensure_worker_locked(reason=f"tool {name!r}")
                    assert self._proxy is not None
                    kw = dict(arguments or {})
                    sink = (
                        event_sink or self._cfg.event_sink_name or ""
                    ).strip() or None
                    kw["_event_sink"] = sink
                    caller = getattr(self._proxy, name, None)
                    if caller is not None:
                        return await caller(**kw)
                    return await self._proxy.as_any().call_tool(
                        name, kw, event_sink=sink
                    )
            except BaseException as exc:
                last_exc = exc
                logger.warning(
                    "supervised worker tool %s failed (attempt %s): %s",
                    name,
                    attempt + 1,
                    exc,
                )
                async with self._lock:
                    await self._teardown_locked()
        raise RuntimeError(f"supervised worker failed after retry: {last_exc!r}")

    async def _ensure_worker_locked(self, *, reason: str) -> None:
        if self._spawn is not None and self._spawn.process.returncode is None:
            return
        await self._teardown_locked()
        self._pump.start()
        logger.info("supervisor spawning ToolWorkerActor (%s)", reason)
        actor = ToolWorkerActor(self._cfg)
        h = await pul.spawn(
            actor,
            new_process=True,
            name="pulsing_forge_worker",
            public=False,
            restart_policy="never",
        )
        if not isinstance(h, pul.IsolatedSpawnHandle):
            raise TypeError("expected IsolatedSpawnHandle from isolated spawn")
        self._spawn = h
        self._proxy = pul.ActorProxy(
            h.ref, ToolWorkerActor._methods, ToolWorkerActor._async_methods
        )

    async def _teardown_locked(self) -> None:
        if self._spawn is None:
            self._proxy = None
            return
        proc = self._spawn.process
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        self._spawn = None
        self._proxy = None
