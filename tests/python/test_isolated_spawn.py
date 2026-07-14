"""Tests for ``spawn(..., new_process=True)`` with a real actor (isolated worker + bridge)."""

from __future__ import annotations

import asyncio

import pytest

import pulsing as pul
from pulsing.core.isolated_bridge import IsolatedSpawnHandle
from pulsing.core.isolated_spawn import spawn_isolated_actor
from pulsing.core.proxy import ActorProxy
from pulsing.core.remote import _extract_methods
from pulsing.testing.isolated_fixtures import IsoMathActor


@pytest.fixture
async def actor_system_addr():
    """Dedicated ActorSystem (non-global) with a real bind address for Connect."""
    system = await pul.actor_system(addr="127.0.0.1:0")
    try:
        yield system
    finally:
        await system.shutdown()


async def _terminate_handle(handle: IsolatedSpawnHandle) -> None:
    if handle.process.returncode is None:
        handle.process.terminate()
        try:
            await asyncio.wait_for(handle.process.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            handle.process.kill()
            await handle.process.wait()


@pytest.mark.asyncio
async def test_isolated_spawn_returns_handle(actor_system_addr):
    handle = await actor_system_addr.spawn(
        IsoMathActor(),
        new_process=True,
        name="test_iso_handle",
        public=True,
        restart_policy="never",
    )
    assert isinstance(handle, IsolatedSpawnHandle)
    assert handle.ref is not None
    assert handle.process.returncode is None
    await _terminate_handle(handle)


@pytest.mark.asyncio
async def test_isolated_spawn_method_roundtrip(actor_system_addr):
    handle = await actor_system_addr.spawn(
        IsoMathActor(),
        new_process=True,
        name="test_iso_roundtrip",
        public=True,
        restart_policy="never",
    )
    try:
        methods, async_methods = _extract_methods(IsoMathActor)
        proxy = ActorProxy(handle.ref, methods, async_methods)
        assert await proxy.mul(6, 7) == 42
    finally:
        await _terminate_handle(handle)


@pytest.mark.asyncio
async def test_isolated_spawn_public_resolve(actor_system_addr):
    """Cluster sees one named actor (the bridge); resolve by name reaches child logic."""
    handle = await actor_system_addr.spawn(
        IsoMathActor(),
        new_process=True,
        name="test_iso_resolve",
        public=True,
        restart_policy="never",
    )
    try:
        ref = await actor_system_addr.resolve("test_iso_resolve")
        methods, async_methods = _extract_methods(IsoMathActor)
        proxy = ActorProxy(ref, methods, async_methods)
        assert await proxy.mul(2, 5) == 10
    finally:
        await _terminate_handle(handle)


@pytest.mark.asyncio
async def test_isolated_spawn_rejects_non_never_restart(actor_system_addr):
    with pytest.raises(ValueError, match="restart_policy"):
        await actor_system_addr.spawn(
            IsoMathActor(),
            new_process=True,
            name="test_iso_restart",
            public=True,
            restart_policy="on_failure",
        )


@pytest.mark.asyncio
async def test_spawn_isolated_actor_inner_api(actor_system_addr):
    """Exercise ``spawn_isolated_actor`` directly (same path as ``ActorSystem.spawn``)."""
    handle = await spawn_isolated_actor(
        actor_system_addr._inner,
        IsoMathActor(),
        name="test_iso_inner",
        public=True,
        restart_policy="never",
        max_restarts=3,
        min_backoff=0.1,
        max_backoff=30.0,
    )
    try:
        methods, async_methods = _extract_methods(IsoMathActor)
        proxy = ActorProxy(handle.ref, methods, async_methods)
        assert await proxy.mul(3, 11) == 33
    finally:
        await _terminate_handle(handle)


@pytest.mark.asyncio
async def test_global_pul_spawn_isolated():
    """``pul.spawn`` + ``pul.init`` dispatch for isolated actors."""
    await pul.init(addr="127.0.0.1:0")
    handle: IsolatedSpawnHandle | None = None
    try:
        out = await pul.spawn(
            IsoMathActor(),
            new_process=True,
            name="test_iso_global",
            public=True,
            restart_policy="never",
        )
        assert isinstance(out, IsolatedSpawnHandle)
        handle = out
        methods, async_methods = _extract_methods(IsoMathActor)
        proxy = ActorProxy(handle.ref, methods, async_methods)
        assert await proxy.mul(8, 8) == 64
    finally:
        if handle is not None:
            await _terminate_handle(handle)
        await pul.shutdown()


@pytest.mark.asyncio
async def test_wait_child_ready_rejects_bad_banner():
    from pulsing.core.isolated_bridge import wait_child_ready

    async def bad_server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        writer.write(b"NOT_READY\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(bad_server, "127.0.0.1", 0)
    try:
        assert server.sockets
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            with pytest.raises(RuntimeError, match="READY"):
                await wait_child_ready(reader)
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()
