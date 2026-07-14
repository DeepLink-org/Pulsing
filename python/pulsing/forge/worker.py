# SPDX-License-Identifier: Apache-2.0
"""Isolated tool worker as a Pulsing actor."""

from __future__ import annotations

from typing import Any

from pulsing.core.remote import remote

from pulsing.forge.approval_bridge import (
    make_worker_exec_approval_callback,
    make_worker_permissions_callback,
)
from pulsing.forge.config import ToolWorkerConfig
from pulsing.forge.context import ToolCallContext
from pulsing.forge.events import ForgeEvent
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.p2p_session import P2PToolSession
from pulsing.forge.p2p_transport import ForgeEventPump, tell_forge_event_sync
from pulsing.forge.hybrid_runtime import HybridForgeRuntime
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.session import NullToolSession

_ISOLATED_ACTOR_NAME = "pulsing_forge_worker"


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


@remote
class ToolWorkerActor:
    """Filesystem/shell tools; returns picklable dicts for RPC."""

    def __init__(self, config: ToolWorkerConfig | None = None) -> None:
        self._cfg = config or ToolWorkerConfig()
        self._pump: ForgeEventPump | None = None
        self._hybrid: HybridForgeRuntime | None = None
        self._runtime: LocalToolRuntime | None = None

    async def on_start(self, actor_id) -> None:
        self._pump = ForgeEventPump(self._cfg.event_sink_name)
        self._pump.start()
        self._runtime = LocalToolRuntime(
            cwd=self._cfg.cwd,
            sandbox_policy=self._cfg.sandbox_policy,
            dangerously_disable_sandbox=self._cfg.dangerously_disable_sandbox,
            session=NullToolSession(),
        )
        if RUST_FORGE_AVAILABLE:
            cb = _event_callback_for_sink(
                self._cfg.event_sink_name,
                self._pump,
                self._cfg.event_sink_name,
            )
            sink = self._cfg.event_sink_name
            approval = self._cfg.approval_sink()
            self._hybrid = HybridForgeRuntime.create(
                cwd=self._cfg.cwd,
                sandbox_policy=self._cfg.sandbox_policy,
                dangerously_disable_sandbox=self._cfg.dangerously_disable_sandbox,
                auto_approve=self._cfg.auto_approve,
                session=NullToolSession(),
                event_callback=cb,
                exec_approval_callback=make_worker_exec_approval_callback(approval),
                request_permissions_callback=make_worker_permissions_callback(approval),
            )

    async def on_stop(self) -> None:
        if self._hybrid is not None:
            self._hybrid.close()
        elif self._runtime is not None:
            self._runtime.close()
        if self._pump is not None:
            await self._pump.stop()

    def ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "kind": "tool_worker",
            "rust_forge": self._hybrid is not None
            and self._hybrid.rust_runtime is not None,
        }

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        event_sink: str | None = None,
    ) -> dict[str, Any]:
        sink = (event_sink or self._cfg.event_sink_name or "").strip() or None
        if self._hybrid is not None:
            return self._hybrid.call_tool(name, arguments).to_dict()

        emit = self._emit_fn(sink)
        session = P2PToolSession(emit=emit)
        if emit is not None:
            emit(ForgeEvent.tool_begin(name, arguments))
        out = self._dispatch(name, arguments, session=session)
        if emit is not None:
            emit(
                ForgeEvent.tool_end(
                    name,
                    is_error=bool(out.is_error),
                    content_preview=out.content,
                )
            )
        return out.to_dict()

    def _emit_fn(self, sink: str | None):
        if not sink or self._pump is None:
            return None
        if sink == self._cfg.event_sink_name and self._pump.enabled:
            return self._pump.emit_sync
        return lambda event: tell_forge_event_sync(sink, event)

    def _dispatch(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        *,
        session: P2PToolSession,
    ):
        if self._runtime is None:
            # on_start hasn't run yet (direct construction, or a message racing
            # startup) — build the local runtime lazily instead of crashing.
            self._runtime = LocalToolRuntime(
                cwd=self._cfg.cwd,
                sandbox_policy=self._cfg.sandbox_policy,
                dangerously_disable_sandbox=self._cfg.dangerously_disable_sandbox,
                session=NullToolSession(),
            )
        ctx = ToolCallContext(
            cwd=self._runtime.context.cwd,
            sandbox_policy=self._runtime.context.sandbox_policy,
            dangerously_disable_sandbox=self._runtime.context.dangerously_disable_sandbox,
            session=session,
            exec=self._runtime.context.exec,
        )
        return dispatch_tool(name, dict(arguments or {}), ctx=ctx)

    def _tool(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        sink = kwargs.pop("_event_sink", None)
        return self.call_tool(name, kwargs, event_sink=sink)

    def Read(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("Read", kwargs)

    def Glob(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("Glob", kwargs)

    def Grep(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("Grep", kwargs)

    def Edit(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("Edit", kwargs)

    def Write(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("Write", kwargs)

    def Bash(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("Bash", kwargs)

    def shell_command(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("shell_command", kwargs)

    def exec_command(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("exec_command", kwargs)

    def write_stdin(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("write_stdin", kwargs)

    def apply_patch(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("apply_patch", kwargs)

    def view_image(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("view_image", kwargs)

    def update_plan(self, **kwargs: Any) -> dict[str, Any]:
        return self._tool("update_plan", kwargs)


def default_worker_name(workspace_id: str) -> str:
    from pulsing.forge.naming import shared_tool_worker_name

    return shared_tool_worker_name(workspace_id)
