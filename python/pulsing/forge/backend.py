# SPDX-License-Identifier: Apache-2.0
"""Forge deployment backends on Pulsing Actor (local host + isolated/remote worker)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import pulsing as pul

from pulsing.forge.approval_bridge import (
    make_worker_exec_approval_callback,
    make_worker_permissions_callback,
)
from pulsing.forge.config import ToolWorkerConfig
from pulsing.forge.events import ForgeEvent
from pulsing.forge.hybrid_runtime import HybridForgeRuntime
from pulsing.forge.integrated import FORGE_HOST_TOOL_NAMES, FORGE_ISOLATED_TOOL_NAMES
from pulsing.forge.mcp.naming import is_mcp_dynamic_tool
from pulsing.forge.naming import shared_tool_worker_name
from pulsing.forge.p2p_transport import ForgeEventPump, tell_forge_event_sync
from pulsing.forge.result import ToolResult
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.session import LocalToolSession, ToolSession
from pulsing.forge.worker import ToolWorkerActor
from pulsing.forge.worker_supervisor import ForgeWorkerSupervisor

logger = logging.getLogger(__name__)


class ForgeBackendMode(str, Enum):
    """How isolated Forge tools are executed."""

    LOCAL = "local"  # in-process only (no worker)
    DEDICATED = "dedicated"  # private ToolWorkerActor per host (new_process spawn)
    SHARED = "shared"  # resolve gossip-named workspace worker


@dataclass
class ForgeHostConfig:
    cwd: str = "."
    sandbox_policy: str = "off"
    dangerously_disable_sandbox: bool = False
    auto_approve: bool = False
    session: ToolSession | None = None


ForgeHostRuntime = HybridForgeRuntime | LocalToolRuntime


def create_host_runtime(
    cfg: ForgeHostConfig,
    *,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    user_input_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    exec_approval_callback: Callable[[dict[str, Any]], str] | None = None,
    request_permissions_callback: (
        Callable[[dict[str, Any]], dict[str, Any]] | None
    ) = None,
    tokens_remaining_callback: Callable[[], int | None] | None = None,
    plugin_install_callback: Callable[[dict[str, Any]], bool | str] | None = None,
) -> ForgeHostRuntime:
    """In-process Host runtime (Hybrid when ``pulsing._core`` is available)."""
    if RUST_FORGE_AVAILABLE:
        return HybridForgeRuntime.create(
            cwd=cfg.cwd,
            sandbox_policy=cfg.sandbox_policy,
            dangerously_disable_sandbox=cfg.dangerously_disable_sandbox,
            auto_approve=cfg.auto_approve,
            session=cfg.session or LocalToolSession(),
            event_callback=event_callback,
            user_input_callback=user_input_callback,
            exec_approval_callback=exec_approval_callback,
            request_permissions_callback=request_permissions_callback,
            tokens_remaining_callback=tokens_remaining_callback,
            plugin_install_callback=plugin_install_callback,
        )
    return LocalToolRuntime(
        cwd=cfg.cwd,
        sandbox_policy=cfg.sandbox_policy,
        dangerously_disable_sandbox=cfg.dangerously_disable_sandbox,
        session=cfg.session or LocalToolSession(),
    )


def _event_callback_for_sink(
    sink: str | None,
    pump: ForgeEventPump,
    default_sink: str | None,
) -> Any:
    if not sink:
        return None

    def _cb(raw: dict[str, Any]) -> None:
        event = ForgeEvent.from_dict(raw)
        if sink == default_sink and pump.enabled:
            pump.emit_sync(event)
        else:
            tell_forge_event_sync(sink, event)

    return _cb


class ForgeIsolatedWorker:
    """``ToolWorkerActor`` lifecycle via Pulsing spawn / resolve."""

    def __init__(
        self,
        worker_cfg: ToolWorkerConfig,
        *,
        mode: ForgeBackendMode = ForgeBackendMode.DEDICATED,
        remote_name: str | None = None,
    ) -> None:
        if mode == ForgeBackendMode.SHARED and not remote_name:
            raise ValueError("SHARED mode requires remote_name")
        if mode == ForgeBackendMode.LOCAL:
            raise ValueError("LOCAL mode does not use ForgeIsolatedWorker")
        self._cfg = worker_cfg
        self._mode = mode
        self._remote_name = remote_name
        self._pump = ForgeEventPump(worker_cfg.event_sink_name)
        self._spawn: pul.IsolatedSpawnHandle | None = None
        self._proxy: pul.ActorProxy | None = None
        self._supervisor: ForgeWorkerSupervisor | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def dedicated(cls, worker_cfg: ToolWorkerConfig) -> ForgeIsolatedWorker:
        return cls(worker_cfg, mode=ForgeBackendMode.DEDICATED)

    @classmethod
    def shared(
        cls, worker_cfg: ToolWorkerConfig, *, workspace_id: str
    ) -> ForgeIsolatedWorker:
        return cls(
            worker_cfg,
            mode=ForgeBackendMode.SHARED,
            remote_name=shared_tool_worker_name(workspace_id),
        )

    @property
    def mode(self) -> ForgeBackendMode:
        return self._mode

    @property
    def proxy(self) -> pul.ActorProxy | None:
        return self._proxy

    def is_dead(self) -> bool:
        if self._mode == ForgeBackendMode.SHARED:
            return self._proxy is None
        if self._supervisor is not None:
            return self._supervisor.proxy is None
        if self._spawn is None:
            return True
        return self._spawn.process.returncode is not None

    async def ensure_ready(self, *, reason: str = "startup") -> None:
        async with self._lock:
            if self._mode == ForgeBackendMode.SHARED:
                if self._proxy is not None:
                    return
                assert self._remote_name is not None
                logger.info(
                    "resolving shared tool worker %s (%s)", self._remote_name, reason
                )
                self._proxy = await pul.resolve(
                    self._remote_name,
                    cls=ToolWorkerActor,
                    timeout=120.0,
                )
                return
            if not self.is_dead():
                return
            await self._spawn_dedicated_locked(reason=reason)

    async def respawn(self, *, reason: str) -> None:
        async with self._lock:
            if self._mode == ForgeBackendMode.SHARED:
                self._proxy = None
                await self.ensure_ready(reason=reason)
                return
            await self._teardown_spawn_locked()
            await self._spawn_dedicated_locked(reason=reason)

    async def close(self) -> None:
        async with self._lock:
            await self._pump.stop()
            if self._supervisor is not None:
                await self._supervisor.close()
                self._supervisor = None
            await self._teardown_spawn_locked()

    def terminate_process(self) -> None:
        if self._mode == ForgeBackendMode.SHARED:
            return
        if self._supervisor is not None:
            self._supervisor.terminate_process()
            return
        if self._spawn is None:
            return
        proc = self._spawn.process
        if proc.returncode is None:
            proc.terminate()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        event_sink: str | None = None,
    ) -> dict[str, Any]:
        await self.ensure_ready(reason=f"tool {name!r}")
        if self._supervisor is not None:
            return await self._supervisor.call_tool(
                name, arguments, event_sink=event_sink
            )
        assert self._proxy is not None
        kw = dict(arguments or {})
        sink = (event_sink or self._cfg.event_sink_name or "").strip() or None
        kw["_event_sink"] = sink
        caller = getattr(self._proxy, name, None)
        if caller is not None:
            return await caller(**kw)
        return await self._proxy.as_any().call_tool(name, kw, event_sink=sink)

    async def _spawn_dedicated_locked(self, *, reason: str) -> None:
        host = (self._cfg.host_name or "").strip()
        if host:
            logger.info("starting in-process worker supervisor (%s)", reason)
            self._supervisor = ForgeWorkerSupervisor(self._cfg)
            await self._supervisor.ensure_ready(reason=reason)
            self._spawn = None
            self._proxy = self._supervisor.proxy
            return
        await self._pump.start()
        logger.info("spawning ToolWorkerActor (%s)", reason)
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
        self._supervisor = None
        self._proxy = pul.ActorProxy(
            h.ref, ToolWorkerActor._methods, ToolWorkerActor._async_methods
        )

    async def _teardown_spawn_locked(self) -> None:
        if self._supervisor is not None:
            await self._supervisor.close()
            self._supervisor = None
            self._proxy = None
            return
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


async def spawn_shared_tool_worker(
    *,
    workspace_id: str,
    cwd: str,
    sandbox_policy: str = "off",
    dangerously_disable_sandbox: bool = False,
) -> Any:
    """Spawn one public ``ToolWorkerActor`` for a workspace (gossip name)."""
    name = shared_tool_worker_name(workspace_id)
    h = await pul.spawn(
        ToolWorkerActor(
            ToolWorkerConfig(
                cwd=cwd,
                sandbox_policy=sandbox_policy,
                dangerously_disable_sandbox=dangerously_disable_sandbox,
            ),
        ),
        new_process=True,
        name=name,
        public=True,
        restart_policy="never",
    )
    logger.info("shared tool worker spawned at %s", name)
    return h


async def resolve_shared_tool_worker(
    workspace_id: str,
    *,
    timeout: float = 120.0,
) -> pul.ActorProxy:
    name = shared_tool_worker_name(workspace_id)
    return await pul.resolve(name, cls=ToolWorkerActor, timeout=timeout)


@dataclass
class ForgeBackend:
    """Host runtime + optional isolated worker — unified Forge on Pulsing."""

    host: ForgeHostRuntime
    worker: ForgeIsolatedWorker | None = None
    event_sink_name: str | None = None

    @property
    def mode(self) -> ForgeBackendMode:
        if self.worker is None:
            return ForgeBackendMode.LOCAL
        return self.worker.mode

    def refresh_mcp(self) -> None:
        if hasattr(self.host, "refresh_mcp"):
            self.host.refresh_mcp()

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        args = dict(arguments or {})
        if name in FORGE_ISOLATED_TOOL_NAMES:
            if self.worker is None:
                return ToolResult(
                    content="Forge isolated worker is not configured", is_error=True
                )
            last_exc: BaseException | None = None
            for attempt in range(2):
                try:
                    raw = await self.worker.call_tool(
                        name,
                        args,
                        event_sink=self.event_sink_name,
                    )
                    return ToolResult.from_dict(raw)
                except BaseException as e:
                    last_exc = e
                    logger.warning(
                        "isolated tool %s failed (attempt %s): %s", name, attempt + 1, e
                    )
                    if self.worker.mode == ForgeBackendMode.SHARED:
                        async with self.worker._lock:
                            self.worker._proxy = None
                    else:
                        await self.worker.respawn(reason=f"recover after {name!r}")
            return ToolResult(
                content=f"isolated tool failed after retry: {last_exc!r}",
                is_error=True,
            )
        if name in FORGE_HOST_TOOL_NAMES or is_mcp_dynamic_tool(name):
            return await asyncio.to_thread(self.host.call_tool, name, args)
        return ToolResult(content=f"Unknown Forge tool: {name}", is_error=True)

    async def ensure_worker(self, *, reason: str = "startup") -> None:
        if self.worker is not None:
            await self.worker.ensure_ready(reason=reason)

    async def close(self) -> None:
        if hasattr(self.host, "close"):
            self.host.close()
        if self.worker is not None:
            await self.worker.close()
