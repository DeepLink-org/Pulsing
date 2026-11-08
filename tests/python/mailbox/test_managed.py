"""Plugin-managed bootstrap, concurrent startup, reuse and crash recovery."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
from unittest.mock import AsyncMock

import pytest

from pulsing.mailbox.managed import (
    ENDPOINT_FILE,
    _running,
    ensure_host,
    seed_configuration,
)


async def stop_host(root, sig=signal.SIGTERM):
    path = root / ENDPOINT_FILE
    if not path.exists():
        return
    pid = json.loads(path.read_text())["pid"]
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        # The sidecar may already have exited before teardown sends the signal.
        pass
    for _ in range(100):
        if not _running(root):
            # Reap it if this test process was the bootstrapper.
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                # Sidecars launched by an MCP process are not children of pytest.
                pass
            return
        await asyncio.sleep(0.05)
    pytest.fail(f"sidecar {pid} failed to stop")


async def test_concurrent_bootstrap_reuses_host_and_recovers_crash(tmp_path):
    try:
        endpoints = await asyncio.gather(
            *(ensure_host(tmp_path, "127.0.0.1:0") for _ in range(5))
        )
        assert len(set(endpoints)) == 1
        first_pid = json.loads((tmp_path / ENDPOINT_FILE).read_text())["pid"]
        assert await ensure_host(tmp_path, "127.0.0.1:0") == endpoints[0]
        with pytest.raises(ValueError, match="already serves"):
            await ensure_host(tmp_path, "0.0.0.0:0")
        assert _running(tmp_path)  # A conflicting config must not stop its owner.
        await stop_host(tmp_path, signal.SIGKILL)
        assert (tmp_path / ENDPOINT_FILE).exists()  # Stale readiness is not liveness.
        await ensure_host(tmp_path, "127.0.0.1:0")
        assert json.loads((tmp_path / ENDPOINT_FILE).read_text())["pid"] != first_pid
    finally:
        await stop_host(tmp_path)


async def test_startup_error_is_reported_without_orphan(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        addr = f"127.0.0.1:{listener.getsockname()[1]}"
        with pytest.raises(RuntimeError, match="sidecar exited"):
            await ensure_host(tmp_path, addr)
    assert not _running(tmp_path)
    assert not (tmp_path / ENDPOINT_FILE).exists()


async def test_seed_join_failure_does_not_bootstrap_local_network(
    monkeypatch, tmp_path
):
    pytest.importorskip("mcp")
    from pulsing.mailbox import mcp
    from pulsing.mailbox import managed

    bootstrap = AsyncMock()
    monkeypatch.setattr(managed, "ensure_host", bootstrap)
    monkeypatch.setattr(mcp.pul, "init", AsyncMock())
    shutdown = AsyncMock()
    monkeypatch.setattr(mcp.pul, "shutdown", shutdown)
    monkeypatch.setattr(
        mcp.MailboxClient,
        "connect",
        AsyncMock(side_effect=TimeoutError("seed offline")),
    )
    server = mcp.create_server(
        "worker", ["127.0.0.1:1"], managed=True, state_dir=str(tmp_path)
    )
    with pytest.raises(TimeoutError, match="seed offline"):
        async with server.settings.lifespan(server):
            pytest.fail("unreachable seed must not initialize")
    bootstrap.assert_not_awaited()
    shutdown.assert_awaited_once()


@pytest.mark.parametrize(
    "seeds, env_seed, expected",
    [
        ([], "1", (True, [])),
        (["1"], None, (True, [])),
        ([], "10.0.0.1:8100", (False, ["10.0.0.1:8100"])),
        (
            [],
            "10.0.0.1:8100, 10.0.0.2:8100",
            (False, ["10.0.0.1:8100", "10.0.0.2:8100"]),
        ),
        (["10.0.0.2:8100"], "1", (False, ["10.0.0.2:8100"])),
        ([], None, (False, [])),
    ],
)
def test_seed_configuration(seeds, env_seed, expected):
    assert seed_configuration(seeds, env_seed) == expected


@pytest.mark.parametrize("seeds", [["1", "10.0.0.1:8100"], ["1", "1"], [""]])
def test_seed_configuration_rejects_ambiguous_modes(seeds):
    with pytest.raises(ValueError):
        seed_configuration(seeds)
