# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge P2P event transport."""

from __future__ import annotations

from pulsing.forge.events import ForgeEvent, ForgeEventKind
from pulsing.forge.exec_output import ExecOutputDelta, ExecStream
from pulsing.forge.p2p_session import P2PToolSession
from pulsing.forge.runtime import LocalToolRuntime


def test_p2p_session_exec_delta() -> None:
    received: list[ForgeEvent] = []
    session = P2PToolSession(emit=received.append)
    session.on_exec_output_delta(
        ExecOutputDelta(session_id=1, stream=ExecStream.PTY, chunk="hello")
    )
    assert len(received) == 1
    assert received[0].kind == ForgeEventKind.EXEC_OUTPUT_DELTA.value
    assert received[0].payload["chunk"] == "hello"


def test_local_runtime_exec_streams_via_p2p_session(tmp_path) -> None:
    received: list[ForgeEvent] = []
    session = P2PToolSession(emit=received.append)
    rt = LocalToolRuntime(cwd=str(tmp_path), session=session)
    out = rt.call_tool(
        "exec_command",
        {"cmd": "echo stream_p2p", "yield_time_ms": 300, "tty": True},
    )
    assert not out.is_error
    delta_kinds = [
        e.kind for e in received if e.kind == ForgeEventKind.EXEC_OUTPUT_DELTA.value
    ]
    assert delta_kinds
    assert any("stream_p2p" in e.payload.get("chunk", "") for e in received)
