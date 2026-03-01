"""
Tests for Ray-like Async API (llms.binding.md)

Covers:
- pul.init() and pul.shutdown()
- pul.spawn(), pul.refer(), pul.resolve()
- @pul.remote decorator
- Counter.spawn() and Counter.resolve()
"""

import asyncio
import pytest

import pulsing as pul
from pulsing.core import Actor


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def initialized_pul():
    """Initialize global pulsing system for testing."""
    await pul.init()
    yield
    await pul.shutdown()


# ============================================================================
# Test: pul.init() and pul.shutdown()
# ============================================================================


@pytest.mark.asyncio
async def test_init_standalone():
    """Test pul.init() with no parameters (standalone mode)."""
    system = await pul.init()
    assert system is not None
    await pul.shutdown()


@pytest.mark.asyncio
async def test_init_with_addr():
    """Test pul.init() with explicit address."""
    system = await pul.init(addr="127.0.0.1:0")
    assert system is not None
    await pul.shutdown()


@pytest.mark.asyncio
async def test_shutdown():
    """Test pul.shutdown() method."""
    await pul.init()
    # Should not raise
    await pul.shutdown()


# ============================================================================
# Test: pul.spawn()
# ============================================================================


class SimpleActor(Actor):
    """Simple actor for testing spawn."""

    async def receive(self, msg):
        if isinstance(msg, dict) and msg.get("action") == "echo":
            return msg.get("data")
        return msg


@pytest.mark.asyncio
async def test_spawn_anonymous(initialized_pul):
    """Test pul.spawn() without name."""
    ref = await pul.spawn(SimpleActor())
    assert ref is not None
    result = await ref.ask({"action": "echo", "data": "test"})
    assert result == "test"


@pytest.mark.asyncio
async def test_spawn_named(initialized_pul):
    """Test pul.spawn() with name."""
    ref = await pul.spawn(SimpleActor(), name="spawn_test_actor")
    assert ref is not None
    result = await ref.ask("hello")
    assert result == "hello"


@pytest.mark.asyncio
async def test_spawn_public(initialized_pul):
    """Test pul.spawn() with public=True."""
    ref = await pul.spawn(SimpleActor(), name="public_spawn_test", public=True)
    assert ref is not None

    # Should be resolvable
    resolved = await pul.resolve("public_spawn_test")
    assert resolved is not None


# ============================================================================
# Test: pul.refer()
# ============================================================================


@pytest.mark.asyncio
async def test_refer_by_actorid(initialized_pul):
    """Test pul.refer() with ActorId."""
    ref = await pul.spawn(SimpleActor(), name="refer_test")
    actor_id = ref.actor_id

    ref2 = await pul.refer(actor_id)
    assert ref2 is not None
    result = await ref2.ask("refer_test_msg")
    assert result == "refer_test_msg"


@pytest.mark.asyncio
async def test_refer_by_string(initialized_pul):
    """Test pul.refer() with string ActorId."""
    ref = await pul.spawn(SimpleActor(), name="refer_str_test")
    actor_id_str = str(ref.actor_id)

    ref2 = await pul.refer(actor_id_str)
    assert ref2 is not None


# ============================================================================
# Test: pul.resolve()
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_public_actor(initialized_pul):
    """Test pul.resolve() for public actor."""
    await pul.spawn(SimpleActor(), name="resolve_public_test", public=True)

    proxy = await pul.resolve("resolve_public_test")
    assert proxy is not None
    assert isinstance(proxy, pul.ActorProxy)
    result = await proxy.ref.ask("resolved_msg")
    assert result == "resolved_msg"


# ============================================================================
# Test: actorref.ask() and actorref.tell()
# ============================================================================


class CounterActor(Actor):
    """Counter actor for testing ask/tell."""

    def __init__(self):
        self.count = 0

    async def receive(self, msg):
        if isinstance(msg, dict):
            action = msg.get("action")
            if action == "incr":
                self.count += 1
            elif action == "decr":
                self.count -= 1
            elif action == "get":
                return self.count
        return self.count


@pytest.mark.asyncio
async def test_ask_response(initialized_pul):
    """Test actorref.ask() returns response."""
    ref = await pul.spawn(CounterActor())
    result = await ref.ask({"action": "get"})
    assert result == 0


@pytest.mark.asyncio
async def test_tell_no_wait(initialized_pul):
    """Test actorref.tell() doesn't wait for response."""
    ref = await pul.spawn(CounterActor())

    # tell() should return immediately
    await ref.tell({"action": "incr"})
    await ref.tell({"action": "incr"})
    await ref.tell({"action": "incr"})

    # Give time for processing
    await asyncio.sleep(0.1)

    result = await ref.ask({"action": "get"})
    assert result == 3


# ============================================================================
# Test: @pul.remote decorator
# ============================================================================


@pul.remote
class RemoteCounter:
    """Counter using @pul.remote decorator."""

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
async def test_remote_spawn(initialized_pul):
    """Test @pul.remote Counter.spawn()."""
    counter = await RemoteCounter.spawn(init=10)
    assert counter is not None
    result = await counter.get()
    assert result == 10


@pytest.mark.asyncio
async def test_remote_spawn_with_name(initialized_pul):
    """Test @pul.remote Counter.spawn(name=...)."""
    counter = await RemoteCounter.spawn(name="named_counter", init=5)
    assert counter is not None
    result = await counter.get()
    assert result == 5


@pytest.mark.asyncio
async def test_remote_sync_method(initialized_pul):
    """Test calling sync method on @pul.remote class."""
    counter = await RemoteCounter.spawn(init=0)
    result = await counter.incr()
    assert result == 1
    result = await counter.incr()
    assert result == 2


@pytest.mark.asyncio
async def test_remote_async_method(initialized_pul):
    """Test calling async method on @pul.remote class."""
    counter = await RemoteCounter.spawn(init=10)
    result = await counter.decr()
    assert result == 9


@pytest.mark.asyncio
async def test_remote_resolve(initialized_pul):
    """Test Counter.resolve() to get ActorProxy for existing actor."""
    # First spawn a named counter
    await RemoteCounter.spawn(name="resolvable_counter", public=True, init=100)

    # Then resolve it
    proxy = await RemoteCounter.resolve("resolvable_counter")
    assert proxy is not None
    result = await proxy.get()
    assert result == 100


@pytest.mark.asyncio
async def test_remote_resolve_and_call(initialized_pul):
    """Test resolve and call methods on resolved actor."""
    await RemoteCounter.spawn(name="call_counter", public=True, init=50)

    proxy = await RemoteCounter.resolve("call_counter")
    # Call methods on resolved proxy
    result = await proxy.incr()
    assert result == 51
    result = await proxy.decr()
    assert result == 50
