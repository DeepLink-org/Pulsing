# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge P0 Pulsing actors (inbox, supervisor, MCP hub, code registry)."""

from __future__ import annotations

import asyncio
import uuid

import pytest

import pulsing as pul
from pulsing.forge.event_inbox import ForgeEventInbox, ensure_forge_event_inbox
from pulsing.forge.events import ForgeEvent
from pulsing.forge.mcp.hub import McpHubActor, ensure_mcp_hub
from pulsing.forge.naming import (
    code_cell_registry_name,
    forge_event_inbox_name,
    mcp_hub_name,
    worker_supervisor_name,
)
from pulsing.forge.code_mode.registry import ensure_code_cell_registry
from pulsing.forge.config import ToolWorkerConfig
from pulsing.forge.worker_supervisor import ForgeWorkerSupervisor


@pytest.mark.asyncio
async def test_forge_event_inbox_records_and_forwards() -> None:
    host = f"forge_host_{uuid.uuid4().hex[:8]}"
    received: dict[str, list] = {"side": [], "stream": []}

    @pul.remote
    class _Host:
        async def on_forge_side_effect(self, event: dict) -> None:
            received["side"].append(dict(event))

        async def on_forge_stream_event(self, event: dict) -> None:
            received["stream"].append(dict(event))

    await pul.init()
    try:
        await _Host.spawn(name=host, public=True)
        inbox = await ensure_forge_event_inbox(host)
        await asyncio.sleep(0.15)
        await inbox.as_any().tell(
            "on_forge_event",
            ForgeEvent(
                kind="plan_updated",
                payload={"plan": [{"step": "a", "status": "pending"}]},
            ).to_dict(),
        )
        await inbox.as_any().tell(
            "on_forge_event",
            ForgeEvent.exec_output_delta(
                session_id=1, stream="pty", chunk="hi"
            ).to_dict(),
        )
        await asyncio.sleep(0.35)
        events = await inbox.get_forge_events()
        assert len(events) == 2
        assert len(received["side"]) == 1
        assert len(received["stream"]) == 1
        assert received["stream"][0]["kind"] == "exec_output_delta"
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_forge_event_inbox_get_events() -> None:
    host = f"forge_host_{uuid.uuid4().hex[:10]}"
    await pul.init()
    try:
        inbox = await ForgeEventInbox.spawn(host, name=forge_event_inbox_name(host))
        await asyncio.sleep(0.1)
        ev = ForgeEvent.tool_begin("Read", {"file_path": "x"})
        await inbox.as_any().tell("on_forge_event", ev.to_dict())
        await asyncio.sleep(0.15)
        out = await inbox.get_forge_events()
        assert len(out) == 1
        assert out[0]["kind"] == "tool_begin"
    finally:
        await pul.shutdown()


def test_forge_naming_helpers() -> None:
    assert forge_event_inbox_name("craft/ws/ws1/alice") == "craft/ws/ws1/alice/events"
    assert worker_supervisor_name("craft/ws/ws1/alice") == "craft/ws/ws1/alice/worker"
    assert mcp_hub_name("ws1") == "craft/ws/ws1/_mcp_hub"
    assert (
        code_cell_registry_name("craft/ws/ws1/alice") == "craft/ws/ws1/alice/code_cells"
    )


@pytest.mark.asyncio
async def test_mcp_hub_spawn_and_list() -> None:
    ws = f"ws_{uuid.uuid4().hex[:8]}"
    await pul.init()
    try:
        hub = await ensure_mcp_hub(ws, cwd=".")
        await asyncio.sleep(0.1)
        out = await hub.refresh()
        assert out["ok"] is True
        assert isinstance(out["tools"], list)
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_code_cell_registry_exec() -> None:
    host = f"forge_host_{uuid.uuid4().hex[:8]}"

    @pul.remote
    class _ToolHost:
        async def call_tool(self, name: str, kwargs: dict) -> dict:
            if name == "Read":
                return {"content": "file-body", "is_error": False}
            return {"content": f"unknown {name}", "is_error": True}

    await pul.init()
    try:
        await _ToolHost.spawn(name=host, public=True)
        reg = await ensure_code_cell_registry(host)
        await asyncio.sleep(0.15)
        source = 'text("ok")\n'
        out = await reg.execute(source, host_name=host)
        assert out["kind"] in {"result", "yielded"}
        assert str(out["cell_id"]).startswith("cell-")
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_worker_supervisor_spawn() -> None:
    cfg = ToolWorkerConfig(cwd=".")
    await pul.init()
    try:
        sup = ForgeWorkerSupervisor(cfg)
        ping = await sup.ping()
        assert ping.get("ok") is True
        assert ping.get("kind") == "tool_worker"
        await sup.close()
    finally:
        await pul.shutdown()
