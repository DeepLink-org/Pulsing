"""Exercise early-exit paths without launching models or waiting for timeouts."""

from __future__ import annotations

import asyncio
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pulsing.cli.launch import mcp_endpoint, run_child


@pytest.mark.parametrize("error", [None, "HTTP startup failed"])
async def test_endpoint_exit_before_readiness_is_reported(monkeypatch, error):
    pytest.importorskip("mcp")
    import uvicorn
    from mcp.server.fastmcp import FastMCP

    class StoppedServer:
        def __init__(self, config):
            self.started = False
            self.should_exit = False

        async def serve(self, sockets):
            if error:
                raise RuntimeError(error)

    monkeypatch.setattr(uvicorn, "Server", StoppedServer)
    expected = error or "MCP endpoint exited before readiness"
    with pytest.raises(RuntimeError, match=expected):
        async with mcp_endpoint(FastMCP("startup-test"), "test-only-value"):
            pytest.fail("an endpoint that exited must never be reported ready")


async def test_server_failure_kills_and_reaps_unresponsive_child(monkeypatch):
    exited = asyncio.Event()
    signals = []

    class Child:
        returncode = None

        async def wait(self):
            await exited.wait()
            return self.returncode

        def send_signal(self, sig):
            signals.append(sig)
            if sig == signal.SIGKILL:
                self.returncode = -signal.SIGKILL
                exited.set()

    async def grace_period_expired(awaitable, timeout):
        raise TimeoutError("simulated child shutdown timeout")

    child = Child()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=child)
    )
    monkeypatch.setattr(asyncio, "wait_for", grace_period_expired)
    server_done = asyncio.get_running_loop().create_future()
    server_done.set_result(None)
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError, match="companion MCP endpoint stopped"):
        await run_child(["fake-codex"], {}, server_done)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert child.returncode == -signal.SIGKILL
    assert signal.getsignal(signal.SIGTERM) == previous
