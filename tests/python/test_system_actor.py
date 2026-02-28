"""
Tests for SystemActor functionality.

Covers:
- Rust SystemActor (system/core) operations via SystemActorProxy
- Python ActorService (system/python_actor_service) operations via PythonActorServiceProxy
"""

import asyncio
import pytest
import pulsing as pul
from pulsing.core import (
    get_python_actor_service,
    get_system_actor,
    remote,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def system():
    """Create a test ActorSystem."""
    system = await pul.actor_system()
    yield system
    await system.shutdown()


@pytest.fixture
async def sys_proxy(system):
    """Create a SystemActorProxy for the test system."""
    return await get_system_actor(system)


@pytest.fixture
async def service_proxy(system):
    """Create a PythonActorServiceProxy for the test system."""
    return await get_python_actor_service(system)


# ============================================================================
# Test: System Auto-Registration
# ============================================================================


@pytest.mark.asyncio
async def test_system_actor_auto_registered(system):
    """SystemActor should be automatically registered on startup."""
    actors = system.local_actor_names()
    assert "system/core" in actors, "SystemActor should be registered"


@pytest.mark.asyncio
async def test_python_actor_service_auto_registered(system):
    """PythonActorService should be automatically registered on startup."""
    actors = system.local_actor_names()
    assert (
        "system/python_actor_service" in actors
    ), "PythonActorService should be registered"


# ============================================================================
# Test: SystemActorProxy
# ============================================================================


@pytest.mark.asyncio
async def test_get_system_actor_proxy(system):
    """Should be able to get SystemActorProxy."""
    sys_proxy = await get_system_actor(system)
    assert sys_proxy is not None
    assert sys_proxy.ref is not None
    assert sys_proxy.ref.is_local()


# ============================================================================
# Test: Ping via Proxy
# ============================================================================


@pytest.mark.asyncio
async def test_ping_via_proxy(sys_proxy, system):
    """Ping via SystemActorProxy should return Pong with node info."""
    result = await sys_proxy.ping()

    assert result["type"] == "Pong"
    assert "node_id" in result
    assert "timestamp" in result
    # node_id is serialized as string in JSON for u128 precision
    assert int(result["node_id"]) == system.node_id.id


# ============================================================================
# Test: Health Check via Proxy
# ============================================================================


@pytest.mark.asyncio
async def test_health_check_via_proxy(sys_proxy):
    """Health check via SystemActorProxy should return healthy status."""
    result = await sys_proxy.health_check()

    assert result["type"] == "Health"
    assert result["status"] == "healthy"
    assert "actors_count" in result
    assert "uptime_secs" in result


# ============================================================================
# Test: Get Node Info via Proxy
# ============================================================================


@pytest.mark.asyncio
async def test_get_node_info_via_proxy(sys_proxy, system):
    """Should return node information via SystemActorProxy."""
    result = await sys_proxy.get_node_info()

    assert result["type"] == "NodeInfo"
    # node_id is serialized as string in JSON for u128 precision
    assert int(result["node_id"]) == system.node_id.id
    assert "addr" in result
    assert "uptime_secs" in result


@pytest.mark.asyncio
async def test_get_node_info_address_format(sys_proxy):
    """Node address should be in IP:port format."""
    result = await sys_proxy.get_node_info()
    addr = result["addr"]

    # Should contain port separator
    assert ":" in addr


# ============================================================================
# Test: Get Metrics via Proxy
# ============================================================================


@pytest.mark.asyncio
async def test_get_metrics_via_proxy(sys_proxy):
    """Should return system metrics via SystemActorProxy."""
    result = await sys_proxy.get_metrics()

    assert result["type"] == "Metrics"
    assert "actors_count" in result
    assert "messages_total" in result
    assert "actors_created" in result
    assert "actors_stopped" in result
    assert "uptime_secs" in result


@pytest.mark.asyncio
async def test_metrics_message_count_increases(sys_proxy):
    """Message count should increase with each message."""
    # Get initial count
    result1 = await sys_proxy.get_metrics()
    initial_count = result1["messages_total"]

    # Send a few more messages
    await sys_proxy.ping()
    await sys_proxy.ping()

    # Get new count
    result2 = await sys_proxy.get_metrics()
    new_count = result2["messages_total"]

    assert new_count > initial_count


# ============================================================================
# Test: List Actors via Proxy
# ============================================================================


@pytest.mark.asyncio
async def test_list_actors_via_proxy(sys_proxy):
    """Actor list should be empty initially (only system actors)."""
    result = await sys_proxy.list_actors()

    # Should be empty or only contain system actors
    assert isinstance(result, list)


# ============================================================================
# Test: PythonActorServiceProxy
# ============================================================================


@pytest.mark.asyncio
async def test_get_python_actor_service_proxy(system):
    """Should be able to get PythonActorServiceProxy."""
    service_proxy = await get_python_actor_service(system)
    assert service_proxy is not None
    assert service_proxy.ref is not None


@pytest.mark.asyncio
async def test_list_registry_via_proxy(service_proxy):
    """PythonActorServiceProxy should list registered actor classes."""
    classes = await service_proxy.list_registry()

    assert classes is not None
    assert isinstance(classes, list)


# ============================================================================
# Test: @remote with PythonActorServiceProxy
# ============================================================================


@remote
class TestCounter:
    """Test counter class for @remote."""

    def __init__(self, init_value=0):
        self.value = init_value

    def increment(self, n=1):
        self.value += n
        return self.value

    def get_value(self):
        return self.value


@pytest.mark.asyncio
async def test_remote_local_creation(system):
    """@remote should allow local actor creation."""
    counter = await TestCounter.spawn(system=system, init_value=10)

    # Should be able to call methods
    result = await counter.increment(5)
    assert result == 15

    result = await counter.get_value()
    assert result == 15


@pytest.mark.asyncio
async def test_remote_class_registered(service_proxy):
    """@remote decorated class should be registered in global registry."""
    classes = await service_proxy.list_registry()

    # TestCounter should be in the registry
    assert any("TestCounter" in name for name in classes)


# ============================================================================
# Test: Multiple Concurrent Requests via Proxy
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_ping_requests(sys_proxy, system):
    """SystemActor should handle concurrent requests via proxy."""
    tasks = [sys_proxy.ping() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    for result in results:
        assert result["type"] == "Pong"
        # node_id is serialized as string in JSON for u128 precision
        assert int(result["node_id"]) == system.node_id.id


@pytest.mark.asyncio
async def test_concurrent_mixed_requests(sys_proxy):
    """SystemActor should handle mixed concurrent requests via proxy."""
    tasks = [
        sys_proxy.ping(),
        sys_proxy.health_check(),
        sys_proxy.get_node_info(),
        sys_proxy.get_metrics(),
        sys_proxy.list_actors(),
    ]
    results = await asyncio.gather(*tasks)

    assert results[0]["type"] == "Pong"
    assert results[1]["type"] == "Health"
    assert results[2]["type"] == "NodeInfo"
    assert results[3]["type"] == "Metrics"
    assert isinstance(results[4], list)


# ============================================================================
# Test: Uptime via Proxy
# ============================================================================


@pytest.mark.asyncio
async def test_uptime_increases(sys_proxy):
    """Uptime should increase over time."""
    result1 = await sys_proxy.get_node_info()
    uptime1 = result1["uptime_secs"]

    await asyncio.sleep(1.1)

    result2 = await sys_proxy.get_node_info()
    uptime2 = result2["uptime_secs"]

    assert uptime2 >= uptime1


# ============================================================================
# Test: Remote Node Access via Proxy
# ============================================================================


@pytest.mark.asyncio
async def test_get_system_actor_for_remote_node(system):
    """get_system_actor with node_id should work (for cluster scenarios)."""
    # For local testing, use local node's ID
    local_node_id = system.node_id.id

    # This should work even with local node_id
    sys_proxy = await get_system_actor(system, node_id=local_node_id)
    result = await sys_proxy.ping()

    assert result["type"] == "Pong"


@pytest.mark.asyncio
async def test_get_python_actor_service_for_remote_node(system):
    """get_python_actor_service with node_id should work (for cluster scenarios)."""
    # For local testing, use local node's ID
    local_node_id = system.node_id.id

    # This should work even with local node_id
    service_proxy = await get_python_actor_service(system, node_id=local_node_id)
    classes = await service_proxy.list_registry()

    assert isinstance(classes, list)
