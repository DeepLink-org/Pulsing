"""
Tests for SystemActor functionality.

Covers:
- Rust SystemActor (system/core) operations
- Python ActorService (_python_actor_service) operations
- System helper functions (list_actors, get_metrics, etc.)
"""

import asyncio
import pytest
from pulsing.actor import (
    Actor,
    ActorId,
    Message,
    SystemConfig,
    create_actor_system,
    list_actors,
    get_metrics,
    get_node_info,
    health_check,
    ping,
    remote,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def system():
    """Create a test ActorSystem."""
    config = SystemConfig.standalone()
    system = await create_actor_system(config)
    yield system
    await system.shutdown()


# ============================================================================
# Test: System Auto-Registration
# ============================================================================


@pytest.mark.asyncio
async def test_system_actor_auto_registered(system):
    """SystemActor should be automatically registered on startup."""
    actors = system.local_actor_names()
    assert "_system_internal" in actors, "SystemActor should be registered"


@pytest.mark.asyncio
async def test_python_actor_service_auto_registered(system):
    """PythonActorService should be automatically registered on startup."""
    actors = system.local_actor_names()
    assert "_python_actor_service" in actors, "PythonActorService should be registered"


# ============================================================================
# Test: SystemActor Reference
# ============================================================================


@pytest.mark.asyncio
async def test_get_system_actor_reference(system):
    """Should be able to get SystemActor reference."""
    sys_ref = await system.system()
    assert sys_ref is not None
    assert sys_ref.is_local()


# ============================================================================
# Test: Ping
# ============================================================================


@pytest.mark.asyncio
async def test_ping_local(system):
    """Ping should return Pong with node info."""
    result = await ping(system)

    assert result["type"] == "Pong"
    assert "node_id" in result
    assert "timestamp" in result
    assert result["node_id"] == system.node_id.id


@pytest.mark.asyncio
async def test_ping_direct_message(system):
    """Direct ping message to SystemActor."""
    sys_ref = await system.system()
    msg = Message.from_json("SystemMessage", {"type": "Ping"})
    resp = await sys_ref.ask(msg)
    data = resp.to_json()

    assert data["type"] == "Pong"
    assert data["node_id"] == system.node_id.id


# ============================================================================
# Test: Health Check
# ============================================================================


@pytest.mark.asyncio
async def test_health_check(system):
    """Health check should return healthy status."""
    result = await health_check(system)

    assert result["type"] == "Health"
    assert result["status"] == "healthy"
    assert "actors_count" in result
    assert "uptime_secs" in result


@pytest.mark.asyncio
async def test_health_check_direct_message(system):
    """Direct health check message to SystemActor."""
    sys_ref = await system.system()
    msg = Message.from_json("SystemMessage", {"type": "HealthCheck"})
    resp = await sys_ref.ask(msg)
    data = resp.to_json()

    assert data["type"] == "Health"
    assert data["status"] == "healthy"


# ============================================================================
# Test: Get Node Info
# ============================================================================


@pytest.mark.asyncio
async def test_get_node_info(system):
    """Should return node information."""
    result = await get_node_info(system)

    assert result["type"] == "NodeInfo"
    assert result["node_id"] == system.node_id.id
    assert "addr" in result
    assert "uptime_secs" in result


@pytest.mark.asyncio
async def test_get_node_info_address_format(system):
    """Node address should be in IP:port format."""
    result = await get_node_info(system)
    addr = result["addr"]

    # Should contain port separator
    assert ":" in addr


# ============================================================================
# Test: Get Metrics
# ============================================================================


@pytest.mark.asyncio
async def test_get_metrics(system):
    """Should return system metrics."""
    result = await get_metrics(system)

    assert result["type"] == "Metrics"
    assert "actors_count" in result
    assert "messages_total" in result
    assert "actors_created" in result
    assert "actors_stopped" in result
    assert "uptime_secs" in result


@pytest.mark.asyncio
async def test_metrics_message_count_increases(system):
    """Message count should increase with each message."""
    # Get initial count
    result1 = await get_metrics(system)
    initial_count = result1["messages_total"]

    # Send a few more messages
    await ping(system)
    await ping(system)

    # Get new count
    result2 = await get_metrics(system)
    new_count = result2["messages_total"]

    assert new_count > initial_count


# ============================================================================
# Test: List Actors
# ============================================================================


@pytest.mark.asyncio
async def test_list_actors_empty_initially(system):
    """Actor list should be empty initially (only system actors)."""
    result = await list_actors(system)

    # Should be empty or only contain system actors
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_list_actors_direct_message(system):
    """Direct ListActors message to SystemActor."""
    sys_ref = await system.system()
    msg = Message.from_json("SystemMessage", {"type": "ListActors"})
    resp = await sys_ref.ask(msg)
    data = resp.to_json()

    assert data["type"] == "ActorList"
    assert "actors" in data


# ============================================================================
# Test: GetActor
# ============================================================================


@pytest.mark.asyncio
async def test_get_actor_not_found(system):
    """GetActor should return error for non-existent actor."""
    sys_ref = await system.system()
    msg = Message.from_json(
        "SystemMessage", {"type": "GetActor", "name": "nonexistent"}
    )
    resp = await sys_ref.ask(msg)
    data = resp.to_json()

    assert data["type"] == "Error"
    assert "not found" in data["message"].lower()


# ============================================================================
# Test: CreateActor (should fail in pure Rust mode)
# ============================================================================


@pytest.mark.asyncio
async def test_create_actor_not_supported_in_rust(system):
    """CreateActor should return error in pure Rust SystemActor."""
    sys_ref = await system.system()
    msg = Message.from_json(
        "SystemMessage",
        {
            "type": "CreateActor",
            "actor_type": "Counter",
            "name": "test_counter",
            "params": {},
            "public": True,
        },
    )
    resp = await sys_ref.ask(msg)
    data = resp.to_json()

    assert data["type"] == "Error"
    assert "not supported" in data["message"].lower()


# ============================================================================
# Test: PythonActorService
# ============================================================================


@pytest.mark.asyncio
async def test_python_actor_service_list_registry(system):
    """PythonActorService should list registered actor classes."""
    service_ref = await system.resolve_named("_python_actor_service")
    msg = Message.from_json("ListRegistry", {})
    resp = await service_ref.ask(msg)
    data = resp.to_json()

    assert data.get("classes") is not None
    assert isinstance(data["classes"], list)


# ============================================================================
# Test: @remote with PythonActorService
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
    counter = await TestCounter.local(system, init_value=10)

    # Should be able to call methods
    result = await counter.increment(5)
    assert result == 15

    result = await counter.get_value()
    assert result == 15


@pytest.mark.asyncio
async def test_remote_class_registered(system):
    """@remote decorated class should be registered in global registry."""
    service_ref = await system.resolve_named("_python_actor_service")
    msg = Message.from_json("ListRegistry", {})
    resp = await service_ref.ask(msg)
    data = resp.to_json()

    # TestCounter should be in the registry
    class_names = data.get("classes", [])
    assert any("TestCounter" in name for name in class_names)


# ============================================================================
# Test: Multiple Concurrent Requests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_ping_requests(system):
    """SystemActor should handle concurrent requests."""
    tasks = [ping(system) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    for result in results:
        assert result["type"] == "Pong"
        assert result["node_id"] == system.node_id.id


@pytest.mark.asyncio
async def test_concurrent_mixed_requests(system):
    """SystemActor should handle mixed concurrent requests."""
    tasks = [
        ping(system),
        health_check(system),
        get_node_info(system),
        get_metrics(system),
        list_actors(system),
    ]
    results = await asyncio.gather(*tasks)

    assert results[0]["type"] == "Pong"
    assert results[1]["type"] == "Health"
    assert results[2]["type"] == "NodeInfo"
    assert results[3]["type"] == "Metrics"
    assert isinstance(results[4], list)


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_message_type(system):
    """SystemActor should handle invalid message types gracefully."""
    sys_ref = await system.system()
    msg = Message.from_json("SystemMessage", {"type": "InvalidType"})
    resp = await sys_ref.ask(msg)
    data = resp.to_json()

    # Should return error for unknown message type
    assert data["type"] == "Error"


@pytest.mark.asyncio
async def test_malformed_message(system):
    """SystemActor should handle malformed messages gracefully."""
    sys_ref = await system.system()
    # Send a message without proper format
    msg = Message.from_json("BadMessage", {"foo": "bar"})
    resp = await sys_ref.ask(msg)
    data = resp.to_json()

    # Should return error
    assert data["type"] == "Error"


# ============================================================================
# Test: Uptime
# ============================================================================


@pytest.mark.asyncio
async def test_uptime_increases(system):
    """Uptime should increase over time."""
    result1 = await get_node_info(system)
    uptime1 = result1["uptime_secs"]

    await asyncio.sleep(1.1)

    result2 = await get_node_info(system)
    uptime2 = result2["uptime_secs"]

    assert uptime2 >= uptime1
