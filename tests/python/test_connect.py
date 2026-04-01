"""
Tests for out-cluster communication via pulsing.connect.

Verifies that an external connector (not a cluster member) can:
1. Resolve named actors through a gateway node
2. Call sync methods (ask pattern)
3. Call async methods (ask pattern)
4. Stream results from async generators
5. Handle errors propagated from the actor
6. Make concurrent calls through the gateway
7. Spawn actors remotely on the cluster
"""

import asyncio

import pytest
import pulsing as pul
from pulsing.core import remote


# ============================================================================
# Test Actors — deployed inside the cluster
# ============================================================================


@remote
class ConnCalc:
    """Stateful calculator with sync methods."""

    def __init__(self, init_value=0):
        self.value = init_value

    def add(self, n):
        self.value += n
        return self.value

    def get(self):
        return self.value

    def multiply(self, a, b):
        return a * b

    def will_fail(self):
        raise ValueError("intentional error")


@remote
class ConnAsyncService:
    """Service with async methods."""

    async def slow_add(self, a, b):
        await asyncio.sleep(0.01)
        return a + b

    async def greet(self, name):
        await asyncio.sleep(0.01)
        return f"hello {name}"


@remote
class ConnStreamService:
    """Service with async generators for streaming."""

    async def count_up(self, n):
        for i in range(n):
            yield i

    async def echo_stream(self, items):
        for item in items:
            await asyncio.sleep(0.001)
            yield item

    async def partial_fail(self, n):
        for i in range(n):
            if i == 3:
                raise RuntimeError("stream error at 3")
            yield i


# ============================================================================
# Fixture: cluster node + out-cluster connector
# ============================================================================


@pytest.fixture
async def cluster_and_connect():
    """Start an ActorSystem (gateway) with PythonActorService and create a connector."""
    from pulsing._core import ActorSystem, SystemConfig
    from pulsing.core.service import PythonActorService, PYTHON_ACTOR_SERVICE_NAME

    loop = asyncio.get_running_loop()

    config = SystemConfig.with_addr("127.0.0.1:0")
    system = await ActorSystem.create(config, loop)
    gateway_addr = system.addr

    # Start PythonActorService so that remote spawn works
    service = PythonActorService(system)
    await system.spawn(service, name=PYTHON_ACTOR_SERVICE_NAME, public=True)

    await ConnCalc.spawn(system=system, name="services/calc", public=True)
    await ConnAsyncService.spawn(system=system, name="services/async_svc", public=True)
    await ConnStreamService.spawn(
        system=system, name="services/stream_svc", public=True
    )

    await asyncio.sleep(0.1)

    from pulsing.connect import Connect

    conn = await Connect.to(gateway_addr)

    yield system, conn

    await conn.close()
    await system.shutdown()


# ============================================================================
# Sync Method Tests
# ============================================================================


@pytest.mark.asyncio
async def test_connect_sync_multiply(cluster_and_connect):
    """Calling a sync method that returns immediately."""
    _, conn = cluster_and_connect
    calc = await conn.resolve("services/calc")
    result = await calc.multiply(6, 7)
    assert result == 42


@pytest.mark.asyncio
async def test_connect_sync_stateful(cluster_and_connect):
    """Stateful sync method calls preserve actor state across requests."""
    _, conn = cluster_and_connect
    calc = await conn.resolve("services/calc")

    assert await calc.add(10) == 10
    assert await calc.add(5) == 15
    assert await calc.get() == 15


# ============================================================================
# Async Method Tests
# ============================================================================


@pytest.mark.asyncio
async def test_connect_async_add(cluster_and_connect):
    """Calling an async method through the gateway."""
    _, conn = cluster_and_connect
    svc = await conn.resolve("services/async_svc")
    result = await svc.slow_add(3, 4)
    assert result == 7


@pytest.mark.asyncio
async def test_connect_async_string_result(cluster_and_connect):
    """Async method returning a string."""
    _, conn = cluster_and_connect
    svc = await conn.resolve("services/async_svc")
    result = await svc.greet("world")
    assert result == "hello world"


# ============================================================================
# Streaming Tests (async generator via async for)
# ============================================================================


@pytest.mark.asyncio
async def test_connect_stream_count(cluster_and_connect):
    """Streaming integers from an async generator."""
    _, conn = cluster_and_connect
    svc = await conn.resolve("services/stream_svc")

    items = []
    async for item in svc.count_up(5):
        items.append(item)

    assert items == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_connect_stream_echo(cluster_and_connect):
    """Streaming mixed-type items."""
    _, conn = cluster_and_connect
    svc = await conn.resolve("services/stream_svc")

    items = []
    async for item in svc.echo_stream(["a", "b", "c"]):
        items.append(item)

    assert items == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_connect_stream_error_mid_stream(cluster_and_connect):
    """Error mid-stream is propagated to the caller."""
    _, conn = cluster_and_connect
    from pulsing.exceptions import PulsingActorError

    svc = await conn.resolve("services/stream_svc")
    items = []
    with pytest.raises(PulsingActorError, match="stream error at 3"):
        async for item in svc.partial_fail(5):
            items.append(item)

    assert items == [0, 1, 2]


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_connect_method_error(cluster_and_connect):
    """Actor-side ValueError propagates to the connector caller."""
    _, conn = cluster_and_connect
    from pulsing.exceptions import PulsingActorError

    calc = await conn.resolve("services/calc")
    with pytest.raises(PulsingActorError, match="intentional error"):
        await calc.will_fail()


@pytest.mark.asyncio
async def test_connect_resolve_nonexistent(cluster_and_connect):
    """Resolving a non-existent actor raises an error."""
    _, conn = cluster_and_connect

    with pytest.raises(Exception):
        await conn.resolve("services/does_not_exist")


# ============================================================================
# Concurrent Access Tests
# ============================================================================


@pytest.mark.asyncio
async def test_connect_concurrent_calls(cluster_and_connect):
    """Multiple concurrent calls through the gateway all succeed."""
    _, conn = cluster_and_connect
    svc = await conn.resolve("services/async_svc")

    coros = [svc.slow_add(i, i) for i in range(10)]
    results = await asyncio.gather(*coros)

    expected = [i + i for i in range(10)]
    assert sorted(results) == expected


# ============================================================================
# Remote Spawn Tests
# ============================================================================


@pytest.mark.asyncio
async def test_connect_spawn_and_call(cluster_and_connect):
    """Spawn an actor remotely and call its methods."""
    _, conn = cluster_and_connect

    calc = await conn.spawn(ConnCalc, name="services/spawned_calc")

    assert await calc.multiply(3, 4) == 12
    assert await calc.add(10) == 10
    assert await calc.get() == 10


@pytest.mark.asyncio
async def test_connect_spawn_with_args(cluster_and_connect):
    """Spawn with constructor arguments."""
    _, conn = cluster_and_connect

    calc = await conn.spawn(ConnCalc, init_value=100, name="services/calc_with_args")

    assert await calc.get() == 100
    assert await calc.add(50) == 150


@pytest.mark.asyncio
async def test_connect_spawn_async_actor(cluster_and_connect):
    """Spawn an actor with async methods and call them."""
    _, conn = cluster_and_connect

    svc = await conn.spawn(ConnAsyncService, name="services/spawned_async")

    result = await svc.slow_add(10, 20)
    assert result == 30


@pytest.mark.asyncio
async def test_connect_spawn_stream_actor(cluster_and_connect):
    """Spawn an actor with streaming methods and stream from it."""
    _, conn = cluster_and_connect

    svc = await conn.spawn(ConnStreamService, name="services/spawned_stream")

    items = []
    async for item in svc.count_up(4):
        items.append(item)

    assert items == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_connect_spawn_then_resolve(cluster_and_connect):
    """Spawn an actor, then resolve it by name from the same connector."""
    _, conn = cluster_and_connect

    await conn.spawn(ConnCalc, init_value=42, name="services/resolvable")

    resolved = await conn.resolve("services/resolvable")
    assert await resolved.get() == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
