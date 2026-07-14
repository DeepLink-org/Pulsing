# SPDX-License-Identifier: Apache-2.0
"""ToolSession that emits Forge events to a single host sink (P2P, not pub/sub)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pulsing.forge.events import ForgeEvent, ForgeEventKind
from pulsing.forge.session import LocalToolSession, PlanItem, UpdatePlanArgs

EmitFn = Callable[[ForgeEvent], None]


@dataclass
class P2PToolSession(LocalToolSession):
    """Forwards session hooks to one downstream consumer via ``emit``."""

    emit: EmitFn | None = None
    _pending: list[ForgeEvent] = field(default_factory=list, repr=False)

    def _send(self, event: ForgeEvent) -> None:
        self._pending.append(event)
        if self.emit is not None:
            self.emit(event)

    def drain_pending(self) -> list[ForgeEvent]:
        out = list(self._pending)
        self._pending.clear()
        return out

    def update_plan(self, args: UpdatePlanArgs) -> None:
        super().update_plan(args)
        self._send(
            ForgeEvent(
                kind=ForgeEventKind.PLAN_UPDATED.value,
                payload={"plan": [p.to_dict() for p in args.plan]},
            )
        )

    def request_new_context(self) -> None:
        super().request_new_context()
        self._send(ForgeEvent(kind=ForgeEventKind.NEW_CONTEXT.value, payload={}))

    def request_user_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from pulsing.forge.session_input import validate_request_user_input

        validate_request_user_input(arguments)
        self._send(
            ForgeEvent(
                kind=ForgeEventKind.USER_INPUT_REQUEST.value,
                payload=dict(arguments),
            )
        )
        return super().request_user_input(arguments)

    def on_exec_output_delta(self, delta: Any) -> None:
        super().on_exec_output_delta(delta)
        if hasattr(delta, "session_id"):
            session_id = int(delta.session_id)
            stream = getattr(getattr(delta, "stream", None), "value", delta.stream)
            chunk = str(delta.chunk)
        else:
            session_id = int(delta["session_id"])
            stream = str(delta.get("stream", "pty"))
            chunk = str(delta.get("chunk", ""))
        self._send(
            ForgeEvent.exec_output_delta(
                session_id=session_id,
                stream=str(stream),
                chunk=chunk,
            )
        )
