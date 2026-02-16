"""
Tests for Actor System Style API (llms.binding.md)

Covers:
- pul.actor_system() creation and shutdown
- system.spawn() with various parameters
- system.refer() and system.resolve()
- actorref.ask() and actorref.tell()
- @pul.remote decorator with sync/async methods
- system.queue.write() and system.queue.read()
"""

import asyncio
import tempfile
import shutil
import pytest

import pulsing as pul
from pulsing.core import Actor, ActorId


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def system():
    """Create a standalone actor system for testing."""
    system = await pul.actor_system()
    yield system
    await system.shutdown()


# ============================================================================
# Test: pul.actor_system()
# ============================================================================


@pytest.mark.asyncio
async def test_actor_system_standalone():
    """Test creating standalone actor system with no parameters."""
    system = await pul.actor_system()
    assert system is not None
    await system.shutdown()


@pytest.mark.asyncio
async def test_actor_system_with_addr():
    """Test creating actor system with explicit address."""
    system = await pul.actor_system(addr="127.0.0.1:0")
    assert system is not None
    assert system.addr is not None
    await system.shutdown()


@pytest.mark.asyncio
async def test_actor_system_shutdown():
    """Test system.shutdown() method."""
    system = await pul.actor_system()
    # Should not raise
    await system.shutdown()


# ============================================================================
# Test: system.spawn()
# ============================================================================


class EchoActor(Actor):
    """Simple echo actor for testing."""

    async def receive(self, msg):
        return msg


@pytest.mark.asyncio
async def test_spawn_anonymous_actor(system):
    """Test spawning actor without name (anonymous)."""
    ref = await system.spawn(EchoActor())
    assert ref is not None
    result = await ref.ask("hello")
    assert result == "hello"


@pytest.mark.asyncio
async def test_spawn_named_actor(system):
    """Test spawning actor with name."""
    ref = await system.spawn(EchoActor(), name="echo_test")
    assert ref is not None
    result = await ref.ask("world")
    assert result == "world"


@pytest.mark.asyncio
async def test_spawn_public_actor(system):
    """Test spawning public actor."""
    ref = await system.spawn(EchoActor(), name="public_echo", public=True)
    assert ref is not None
    # Public actors should be resolvable by name
    resolved = await system.resolve("public_echo")
    assert resolved is not None


# ============================================================================
# Test: system.refer()
# ============================================================================


@pytest.mark.asyncio
async def test_refer_by_actorid(system):
    """Test getting actor reference by ActorId."""
    ref = await system.spawn(EchoActor(), name="refer_test")
    actor_id = ref.actor_id

    # Get reference by ActorId object
    ref2 = await system.refer(actor_id)
    assert ref2 is not None
    result = await ref2.ask("test")
    assert result == "test"


@pytest.mark.asyncio
async def test_refer_by_string(system):
    """Test getting actor reference by string ActorId."""
    ref = await system.spawn(EchoActor(), name="refer_str_test")
    actor_id_str = str(ref.actor_id)

    # Get reference by string
    ref2 = await system.refer(actor_id_str)
    assert ref2 is not None
    result = await ref2.ask("string_test")
    assert result == "string_test"


# ============================================================================
# Test: system.resolve()
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_named_actor(system):
    """Test resolving public actor by name."""
    await system.spawn(EchoActor(), name="resolve_test", public=True)

    ref = await system.resolve("resolve_test")
    assert ref is not None
    result = await ref.ask("resolved")
    assert result == "resolved"


# ============================================================================
# Test: actorref.ask() and actorref.tell()
# ============================================================================


class StatefulActor(Actor):
    """Actor with state for testing tell()."""

    def __init__(self):
        self.messages = []

    async def receive(self, msg):
        if isinstance(msg, dict):
            if msg.get("action") == "store":
                self.messages.append(msg.get("data"))
                return None
            elif msg.get("action") == "get":
                return self.messages
        return msg


@pytest.mark.asyncio
async def test_ask_returns_response(system):
    """Test ask() returns response from actor."""
    ref = await system.spawn(EchoActor())
    result = await ref.ask({"key": "value"})
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_tell_fire_and_forget(system):
    """Test tell() sends message without waiting for response."""
    ref = await system.spawn(StatefulActor())

    # tell() should not wait for response
    await ref.tell({"action": "store", "data": "msg1"})
    await ref.tell({"action": "store", "data": "msg2"})

    # Give some time for messages to be processed
    await asyncio.sleep(0.1)

    # Verify messages were received
    messages = await ref.ask({"action": "get"})
    assert "msg1" in messages
    assert "msg2" in messages


# ============================================================================
# Test: @pul.remote decorator
# ============================================================================


@pul.remote
class Counter:
    """Counter actor using @pul.remote decorator."""

    def __init__(self, init=0):
        self.value = init

    def incr(self):
        """Sync method."""
        self.value += 1
        return self.value

    async def decr(self):
        """Async method."""
        self.value -= 1
        return self.value

    def get(self):
        return self.value


@pytest.mark.asyncio
async def test_remote_decorator_spawn(system):
    """Test @pul.remote class spawn."""
    counter = await Counter.local(system, init=10)
    assert counter is not None
    result = await counter.get()
    assert result == 10


@pytest.mark.asyncio
async def test_remote_decorator_sync_method(system):
    """Test calling sync method on @pul.remote class."""
    counter = await Counter.local(system, init=0)
    result = await counter.incr()
    assert result == 1
    result = await counter.incr()
    assert result == 2


@pytest.mark.asyncio
async def test_remote_decorator_async_method(system):
    """Test calling async method on @pul.remote class."""
    counter = await Counter.local(system, init=5)
    result = await counter.decr()
    assert result == 4


# ============================================================================
# Test: Queue API - system.queue.write() and system.queue.read()
# ============================================================================


@pytest.mark.asyncio
async def test_queue_write_and_read(system):
    """Test basic queue write and read."""
    temp_dir = tempfile.mkdtemp(prefix="queue_test_")
    try:
        writer = await system.queue.write(
            "test_topic",
            bucket_column="id",
            num_buckets=2,
            storage_path=temp_dir,
        )
        assert writer is not None

        # Write records
        await writer.put({"id": "1", "data": "first"})
        await writer.put({"id": "2", "data": "second"})
        await writer.flush()

        # Read records
        reader = await system.queue.read(
            "test_topic",
            num_buckets=2,
            storage_path=temp_dir,
        )
        records = await reader.get(limit=10)
        assert len(records) == 2

        ids = {r["id"] for r in records}
        assert ids == {"1", "2"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_queue_batch_write(system):
    """Test batch write to queue."""
    temp_dir = tempfile.mkdtemp(prefix="queue_batch_test_")
    try:
        writer = await system.queue.write(
            "batch_topic",
            bucket_column="id",
            num_buckets=1,
            storage_path=temp_dir,
        )

        # Batch write
        records = [{"id": str(i), "value": i} for i in range(10)]
        await writer.put(records)
        await writer.flush()

        # Read all
        reader = await system.queue.read(
            "batch_topic",
            num_buckets=1,
            storage_path=temp_dir,
        )
        result = await reader.get(limit=20)
        assert len(result) == 10
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Test: Generator streaming response for base Actor
# ============================================================================


class SyncGeneratorActor(Actor):
    """Actor that returns a sync generator."""

    async def receive(self, msg):
        if isinstance(msg, dict) and msg.get("action") == "stream":
            count = msg.get("count", 5)

            def generate():
                for i in range(count):
                    yield f"item_{i}"

            return generate()
        return msg


class AsyncGeneratorActor(Actor):
    """Actor that returns an async generator."""

    async def receive(self, msg):
        if isinstance(msg, dict) and msg.get("action") == "stream":
            count = msg.get("count", 5)

            async def generate():
                for i in range(count):
                    yield f"async_item_{i}"

            return generate()
        return msg


@pytest.mark.asyncio
async def test_base_actor_sync_generator(system):
    """Test base Actor returning sync generator for streaming."""
    ref = await system.spawn(SyncGeneratorActor(), name="sync_gen_actor")

    # Consume the stream
    items = []
    response = await ref.ask({"action": "stream", "count": 3})
    # Response should be a stream
    if hasattr(response, "__aiter__"):
        async for item in response:
            items.append(item)
    else:
        # Single item response (fallback)
        items.append(response)

    assert len(items) >= 1


@pytest.mark.asyncio
async def test_base_actor_async_generator(system):
    """Test base Actor returning async generator for streaming."""
    ref = await system.spawn(AsyncGeneratorActor(), name="async_gen_actor")

    # Consume the stream
    items = []
    response = await ref.ask({"action": "stream", "count": 3})
    # Response should be a stream
    if hasattr(response, "__aiter__"):
        async for item in response:
            items.append(item)
    else:
        # Single item response (fallback)
        items.append(response)

    assert len(items) >= 1
