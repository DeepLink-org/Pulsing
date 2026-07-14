# SPDX-License-Identifier: Apache-2.0
"""Tests for Craft forge event handler."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from pulsing.agent.actors.forge_events import emit_forge_event, handle_forge_event
from pulsing.forge.events import ForgeEvent
from pulsing.forge.p2p_transport import tell_forge_event


class _FakeAgent:
    def __init__(self) -> None:
        self._forge_events: list[dict] = []
        self._forge_stream_sink = None
        self._stream_chunks: list[dict] = []

    async def _sink(self, ev: dict) -> None:
        self._stream_chunks.append(ev)


@pytest.mark.asyncio
async def test_handle_forge_event_records_and_streams() -> None:
    agent = _FakeAgent()
    agent._forge_stream_sink = agent._sink
    event = ForgeEvent.exec_output_delta(session_id=2, stream="pty", chunk="x")
    await handle_forge_event(agent, event.to_dict())
    assert len(agent._forge_events) == 1
    assert agent._stream_chunks[0]["kind"] == "forge_exec_delta"


@pytest.mark.asyncio
async def test_tell_forge_event_delivers_to_named_actor() -> None:
    import pulsing as pul

    @pul.remote
    class _ForgeEventSink:
        def __init__(self) -> None:
            self._events: list[dict] = []

        async def on_forge_event(self, event: dict) -> None:
            self._events.append(dict(event))

        def get_forge_events(self) -> list[dict]:
            return list(self._events)

    name = f"forge_sink_{uuid.uuid4().hex[:10]}"
    await pul.init()
    try:
        sink = await _ForgeEventSink.spawn(public=True, name=name)
        await asyncio.sleep(0.2)
        await tell_forge_event(name, ForgeEvent.tool_begin("Read", {"file_path": "x"}))
        await asyncio.sleep(0.2)
        events = await sink.get_forge_events()
        assert len(events) == 1
        assert events[0]["kind"] == "tool_begin"
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_emit_forge_event_helper_uses_tell() -> None:
    agent = _FakeAgent()
    agent._event_sink_name = "craft/ws/test/agent"
    event = ForgeEvent.tool_begin("Test", {"x": 1})

    called: list[tuple[str, str]] = []

    async def _fake_tell(sink: str, ev: ForgeEvent) -> None:
        called.append((sink, ev.kind))
        await handle_forge_event(agent, ev.to_dict())

    import pulsing.agent.actors.forge_events as fe

    orig = fe.tell_forge_event
    fe.tell_forge_event = _fake_tell
    try:
        await emit_forge_event(agent, event)
    finally:
        fe.tell_forge_event = orig

    assert called == [("craft/ws/test/agent", "tool_begin")]
    assert len(agent._forge_events) == 1
