# SPDX-License-Identifier: Apache-2.0
"""Handle Forge P2P events on workspace Agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pulsing.forge.events import ForgeEvent, ForgeEventKind
from pulsing.forge.p2p_transport import tell_forge_event, tell_forge_event_sync

logger = logging.getLogger(__name__)


def forge_event_sink(agent: Any) -> str | None:
    name = getattr(agent, "_event_sink_name", None)
    if not name:
        return None
    return str(name).strip() or None


async def emit_forge_event(agent: Any, event: ForgeEvent) -> None:
    """Deliver a forge event via actor tell (same path as cross-process workers)."""
    sink = forge_event_sink(agent)
    if not sink:
        logger.debug("forge event sink unset; handling locally kind=%s", event.kind)
        await handle_forge_event(agent, event.to_dict())
        return
    await tell_forge_event(sink, event)


def emit_forge_event_sync(agent: Any, event: ForgeEvent) -> None:
    """Sync tell from tool threads / Rust callbacks."""
    sink = forge_event_sink(agent)
    if not sink:
        logger.debug("forge event sink unset; handling locally kind=%s", event.kind)
        _schedule_local(agent, event)
        return
    tell_forge_event_sync(sink, event)


def _schedule_local(agent: Any, event: ForgeEvent) -> None:
    raw = event.to_dict()

    async def _deliver() -> None:
        await handle_forge_event(agent, raw)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_deliver())
    except RuntimeError:
        asyncio.run(_deliver())


def make_host_emit(agent: Any):
    return lambda event: emit_forge_event_sync(agent, event)


async def apply_forge_side_effects(agent: Any, raw: dict[str, Any]) -> None:
    """Update session / activity / stream sink without recording events."""
    event = ForgeEvent.from_dict(raw)

    if event.kind == ForgeEventKind.PLAN_UPDATED.value:
        session = getattr(agent, "_forge_session", None)
        if session is not None:
            from pulsing.forge.session import PlanItem, StepStatus

            items = []
            for item in event.payload.get("plan") or []:
                status = item.get("status", StepStatus.PENDING)
                if isinstance(status, str):
                    status = StepStatus(status)
                items.append(PlanItem(step=str(item.get("step", "")), status=status))
            session.plan = items

    if event.kind == ForgeEventKind.NEW_CONTEXT.value:
        session = getattr(agent, "_forge_session", None)
        if session is not None:
            session.new_context_requested = True

    sink = getattr(agent, "_forge_stream_sink", None)
    if sink is not None:
        payload = _to_stream_chunk(event)
        if payload is not None:
            result = sink(payload)
            if hasattr(result, "__await__"):
                await result

    if event.kind == ForgeEventKind.EXEC_OUTPUT_DELTA.value and hasattr(
        agent, "_activity"
    ):
        chunk = str(event.payload.get("chunk") or "")
        if chunk:
            detail = chunk.strip().splitlines()[-1][:120]
            from pulsing.agent.actors.activity import set_activity

            set_activity(
                agent,
                state="tool",
                detail=detail or "exec…",
                tool="exec_command",
            )


async def handle_forge_event(agent: Any, raw: dict[str, Any]) -> None:
    events = getattr(agent, "_forge_events", None)
    if events is None:
        agent._forge_events = []
        events = agent._forge_events
    events.append(dict(raw))
    await apply_forge_side_effects(agent, raw)


def _to_stream_chunk(event: ForgeEvent) -> dict[str, Any] | None:
    if event.kind == ForgeEventKind.EXEC_OUTPUT_DELTA.value:
        return {
            "kind": "forge_exec_delta",
            "session_id": event.payload.get("session_id"),
            "stream": event.payload.get("stream"),
            "chunk": event.payload.get("chunk"),
        }
    if event.kind == ForgeEventKind.TOOL_BEGIN.value:
        return {"kind": "forge_tool_begin", "tool": event.source, **event.payload}
    if event.kind == ForgeEventKind.TOOL_END.value:
        return {"kind": "forge_tool_end", "tool": event.source, **event.payload}
    if event.kind == ForgeEventKind.PLAN_UPDATED.value:
        return {"kind": "forge_plan_updated", **event.payload}
    if event.kind == ForgeEventKind.NEW_CONTEXT.value:
        return {"kind": "forge_new_context"}
    if event.kind == ForgeEventKind.USER_INPUT_REQUEST.value:
        return {"kind": "forge_user_input_request", **event.payload}
    return {"kind": "forge_event", "forge_kind": event.kind, **event.payload}
