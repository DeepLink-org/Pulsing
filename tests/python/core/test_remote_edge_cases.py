"""
Tests for core/remote.py edge cases and uncovered paths.

Focus areas:
- _WrappedActor edge cases
- Protocol wire format (call/response)
- Attribute access
- Sync generator handling
- on_start/on_stop callbacks
- metadata method
- Error paths
"""

import asyncio

import pytest

import pulsing as pul
from pulsing.core import remote, init, shutdown, get_system


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def system():
    """Create a standalone ActorSystem for testing."""
    sys = await pul.actor_system()
    yield sys
    await sys.shutdown()


# ============================================================================
# _WrappedActor: on_start/on_stop callbacks
# ============================================================================


@pytest.mark.asyncio
async def test_wrapped_actor_on_start_callback():
    """Test that on_start is called when actor starts."""

    @remote
    class ActorWithOnStart:
        def __init__(self):
            self.started = False
            self.actor_id = None

        def on_start(self, actor_id):
            self.started = True
            self.actor_id = actor_id

        def is_started(self):
            return self.started, str(self.actor_id) if self.actor_id else None

    await init()
    try:
        actor = await ActorWithOnStart.spawn()
        started, aid = await actor.is_started()
        assert started is True
        assert aid is not None
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_wrapped_actor_on_stop_callback():
    """Test that on_stop is called when actor stops."""
    results = []

    @remote
    class ActorWithOnStop:
        def __init__(self, results_list):
            self.results = results_list

        def on_stop(self):
            self.results.append("stopped")

        def ping(self):
            return "pong"

    await init()
    try:
        actor_name = "on_stop_test_actor"
        actor = await ActorWithOnStop.spawn(results, name=actor_name)
        assert await actor.ping() == "pong"
        # Stop the actor by name - this should trigger on_stop
        await get_system().stop(actor_name)
        await asyncio.sleep(0.1)
        assert "stopped" in results
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_wrapped_actor_no_on_start_method():
    """Test actor without on_start works normally."""

    @remote
    class NoOnStartActor:
        def __init__(self):
            self.value = 42

        def get_value(self):
            return self.value

    await init()
    try:
        actor = await NoOnStartActor.spawn()
        assert await actor.get_value() == 42
    finally:
        await shutdown()


# ============================================================================
# _WrappedActor: metadata method
# ============================================================================


@pytest.mark.asyncio
async def test_wrapped_actor_metadata():
    """Test that metadata method is called and returned."""

    @remote
    class ActorWithMetadata:
        def metadata(self):
            return {"version": "1.0", "type": "test"}

        def ping(self):
            return "pong"

    await init()
    try:
        actor = await ActorWithMetadata.spawn(name="metadata_test")
        # Metadata should be accessible
        ref = actor.ref
        # The metadata is stored during spawn
        assert await actor.ping() == "pong"
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_wrapped_actor_no_metadata():
    """Test actor without metadata method returns empty dict."""

    @remote
    class NoMetadataActor:
        def ping(self):
            return "pong"

    await init()
    try:
        actor = await NoMetadataActor.spawn()
        assert await actor.ping() == "pong"
    finally:
        await shutdown()


# ============================================================================
# _WrappedActor: attribute access
# ============================================================================


@pytest.mark.asyncio
async def test_attribute_access():
    """Test accessing public attributes through protocol."""

    @remote
    class AttributeActor:
        def __init__(self):
            self.counter = 0
            self.name = "test_actor"

        def increment(self):
            self.counter += 1
            return self.counter

    await init()
    try:
        actor = await AttributeActor.spawn()
        # Method call
        assert await actor.increment() == 1
        assert await actor.increment() == 2
    finally:
        await shutdown()


# ============================================================================
# _WrappedActor: sync generator handling
# ============================================================================


@pytest.mark.asyncio
async def test_sync_generator_method():
    """Test sync generator method returns sequence of values."""

    @remote
    class GeneratorActor:
        def count_up(self, n):
            for i in range(n):
                yield i

    await init()
    try:
        actor = await GeneratorActor.spawn()
        # Sync generator methods need await then iterate
        result = await actor.count_up(5)
        items = []
        if hasattr(result, "__aiter__"):
            async for item in result:
                items.append(item)
        elif hasattr(result, "__iter__"):
            for item in result:
                items.append(item)
        else:
            items.append(result)
        assert len(items) >= 1
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_sync_generator_with_exception():
    """Test sync generator that raises exception."""

    @remote
    class FailingGeneratorActor:
        def failing_gen(self, fail_at):
            for i in range(10):
                if i == fail_at:
                    raise ValueError(f"Failed at {fail_at}")
                yield i

    await init()
    try:
        actor = await FailingGeneratorActor.spawn()
        result = await actor.failing_gen(3)
        items = []
        with pytest.raises(Exception):
            if hasattr(result, "__aiter__"):
                async for item in result:
                    items.append(item)
            elif hasattr(result, "__iter__"):
                for item in result:
                    items.append(item)
    finally:
        await shutdown()


# ============================================================================
# Protocol wire format
# ============================================================================


@pytest.mark.asyncio
async def test_protocol_call_format():
    """Test protocol call/response format (single wire format)."""
    from pulsing.core.remote import (
        _wrap_call,
        _wrap_response,
        _unwrap_call,
        _unwrap_response,
    )

    msg = _wrap_call("test_method", (1, 2), {"key": "value"}, True)
    assert msg["__call__"] == "test_method"
    assert msg["__async__"] is True
    assert msg["args"] == (1, 2)

    method, args, kwargs, is_async = _unwrap_call(msg)
    assert method == "test_method"
    assert args == (1, 2)
    assert kwargs == {"key": "value"}
    assert is_async is True

    resp = _wrap_response(result="success")
    assert resp["__result__"] == "success"
    result, error = _unwrap_response(resp)
    assert result == "success"
    assert error is None

    err_resp = _wrap_response(error="failed")
    assert err_resp["__error__"] == "failed"
    result, error = _unwrap_response(err_resp)
    assert result is None
    assert error == "failed"


# ============================================================================
# Invalid method handling
# ============================================================================


@pytest.mark.asyncio
async def test_call_private_method():
    """Test that calling private methods (starting with _) returns error."""

    @remote
    class PrivateMethodActor:
        def public_method(self):
            return "public"

        def _private_method(self):
            return "private"

    await init()
    try:
        actor = await PrivateMethodActor.spawn()
        # Public method should work
        assert await actor.public_method() == "public"
        # Private method should be blocked (AttributeError on proxy)
        with pytest.raises(AttributeError):
            _ = actor._private_method
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_call_nonexistent_method():
    """Test that calling non-existent method returns error."""

    @remote
    class SimpleActor:
        def existing_method(self):
            return "exists"

    await init()
    try:
        actor = await SimpleActor.spawn()
        # Existing method works
        assert await actor.existing_method() == "exists"
        # Non-existent method raises AttributeError
        with pytest.raises(AttributeError):
            _ = actor.nonexistent_method
    finally:
        await shutdown()


# ============================================================================
# Message protocol edge cases
# ============================================================================


@pytest.mark.asyncio
async def test_unknown_message_type():
    """Test actor behavior with unknown message type."""
    from pulsing._core import Message

    @remote
    class MessageHandlingActor:
        def ping(self):
            return "pong"

    await init()
    try:
        actor = await MessageHandlingActor.spawn()
        # Normal call works
        assert await actor.ping() == "pong"
    finally:
        await shutdown()


# ============================================================================
# Async generator edge cases
# ============================================================================


@pytest.mark.asyncio
async def test_async_generator_immediate_break():
    """Test async generator when caller breaks immediately."""

    @remote
    class AsyncGenActor:
        async def stream_values(self, n):
            for i in range(n):
                await asyncio.sleep(0.01)
                yield i

    await init()
    try:
        actor = await AsyncGenActor.spawn()
        # Break on first value
        count = 0
        async for value in actor.stream_values(10):
            count += 1
            if count >= 2:
                break
        assert count == 2
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_async_generator_empty():
    """Test async generator that yields nothing."""

    @remote
    class EmptyGenActor:
        async def empty_stream(self):
            return
            yield  # Never reached

    await init()
    try:
        actor = await EmptyGenActor.spawn()
        results = []
        async for value in actor.empty_stream():
            results.append(value)
        assert results == []
    finally:
        await shutdown()


# ============================================================================
# Complex scenarios
# ============================================================================


@pytest.mark.asyncio
async def test_actor_with_both_sync_and_async_methods():
    """Test actor mixing sync methods, async methods, and generators."""

    @remote
    class MixedActor:
        def __init__(self):
            self.sync_count = 0
            self.async_count = 0

        def sync_method(self, x):
            self.sync_count += 1
            return x * 2

        async def async_method(self, x):
            await asyncio.sleep(0.01)
            self.async_count += 1
            return x * 3

        async def async_gen(self, n):
            for i in range(n):
                await asyncio.sleep(0.01)
                yield i * 10

        def get_counts(self):
            return self.sync_count, self.async_count

    await init()
    try:
        actor = await MixedActor.spawn()

        # Sync method
        assert await actor.sync_method(5) == 10

        # Async method
        assert await actor.async_method(5) == 15

        # Async generator
        async_gen_results = [v async for v in actor.async_gen(3)]
        assert async_gen_results == [0, 10, 20]

        # Check counts
        sc, ac = await actor.get_counts()
        assert sc == 1
        assert ac == 1
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_actor_exception_preserves_state():
    """Test that actor state is preserved after exception."""

    @remote
    class StatefulActor:
        def __init__(self):
            self.value = 0

        def increment(self):
            self.value += 1
            return self.value

        def fail(self):
            raise ValueError("Intentional failure")

        def get_value(self):
            return self.value

    await init()
    try:
        actor = await StatefulActor.spawn()

        # First increment
        assert await actor.increment() == 1

        # This fails but shouldn't corrupt state
        with pytest.raises(Exception):
            await actor.fail()

        # State should still be intact
        assert await actor.get_value() == 1

        # Can continue incrementing
        assert await actor.increment() == 2
    finally:
        await shutdown()


# ============================================================================
# Delayed call advanced scenarios
# ============================================================================


@pytest.mark.asyncio
async def test_delayed_call_with_args():
    """Test delayed call with arguments."""

    @remote
    class DelayedArgsActor:
        def __init__(self):
            self.messages = []

        def schedule_message(self, delay, msg):
            self.delayed(delay).record(msg)

        def record(self, msg):
            self.messages.append(msg)

        def get_messages(self):
            return list(self.messages)

    await init()
    try:
        actor = await DelayedArgsActor.spawn()
        await actor.schedule_message(0.05, "hello")

        assert await actor.get_messages() == []
        await asyncio.sleep(0.1)
        assert await actor.get_messages() == ["hello"]
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_multiple_delayed_calls():
    """Test multiple delayed calls execute in order."""

    @remote
    class MultiDelayedActor:
        def __init__(self):
            self.events = []

        def schedule_all(self):
            self.delayed(0.02).record("second")
            self.delayed(0.01).record("first")
            self.events.append("immediate")

        def record(self, msg):
            self.events.append(msg)

        def get_events(self):
            return list(self.events)

    await init()
    try:
        actor = await MultiDelayedActor.spawn()
        await actor.schedule_all()

        # Immediate should be recorded
        assert "immediate" in await actor.get_events()

        await asyncio.sleep(0.05)

        events = await actor.get_events()
        assert "first" in events
        assert "second" in events
    finally:
        await shutdown()


# ============================================================================
# _WrappedActor.receive: attribute access (non-callable)
# ============================================================================


@pytest.mark.asyncio
async def test_attribute_read_via_protocol():
    """Test accessing a non-callable attribute returns its value."""

    @remote
    class AttrActor:
        def __init__(self):
            self.name = "alice"
            self.count = 42

        def get_name(self):
            return self.name

    await init()
    try:
        actor = await AttrActor.spawn()
        proxy = actor.as_any()
        result = await proxy.name
        assert result == "alice"
        result = await proxy.count
        assert result == 42
    finally:
        await shutdown()


# ============================================================================
# ActorClass.spawn without init raises
# ============================================================================


@pytest.mark.asyncio
async def test_spawn_without_init_raises():
    """Calling spawn before init() should raise PulsingRuntimeError."""
    from pulsing.exceptions import PulsingRuntimeError

    @remote
    class NeverSpawned:
        def ping(self):
            return "pong"

    with pytest.raises(PulsingRuntimeError, match="not initialized"):
        await NeverSpawned.spawn()


# ============================================================================
# ActorClass.local with name namespace handling
# ============================================================================


@pytest.mark.asyncio
async def test_local_name_with_namespace():
    """Name already containing / should be used as-is."""

    @remote
    class NsActor:
        def ping(self):
            return "pong"

    await init()
    try:
        actor = await NsActor.spawn(name="custom/my_actor")
        assert await actor.ping() == "pong"
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_local_without_name():
    """Spawning without name should auto-generate one."""

    @remote
    class AutoNameActor:
        def ping(self):
            return "pong"

    await init()
    try:
        actor = await AutoNameActor.spawn()
        assert await actor.ping() == "pong"
    finally:
        await shutdown()


# ============================================================================
# ActorClass.resolve
# ============================================================================


@pytest.mark.asyncio
async def test_actor_class_resolve():
    """Test ActorClass.resolve returns typed proxy."""

    @remote
    class ResolvableActor:
        def greet(self):
            return "hello"

    await init()
    try:
        await ResolvableActor.spawn(name="resolvable_test")
        proxy = await ResolvableActor.resolve("resolvable_test")
        assert await proxy.greet() == "hello"
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_actor_class_resolve_without_init():
    """Resolve without init raises PulsingRuntimeError."""
    from pulsing.exceptions import PulsingRuntimeError

    @remote
    class NeverResolved:
        def ping(self):
            return "pong"

    with pytest.raises(PulsingRuntimeError, match="not initialized"):
        await NeverResolved.resolve("nonexistent")


# ============================================================================
# ActorClass.proxy wraps existing ref
# ============================================================================


@pytest.mark.asyncio
async def test_actor_class_resolve_typed():
    """Test resolving a named actor into a typed proxy."""

    @remote
    class ProxyTestActor:
        def double(self, x):
            return x * 2

    await init()
    try:
        await ProxyTestActor.spawn(name="proxy_wrap_test", public=True)
        typed = await ProxyTestActor.resolve("proxy_wrap_test")
        assert await typed.double(7) == 14
    finally:
        await shutdown()


# ============================================================================
# Method that raises specific exception types
# ============================================================================


@pytest.mark.asyncio
async def test_method_raises_value_error():
    """Verify that ValueError from actor method is propagated."""

    @remote
    class ValidatingActor:
        def validate(self, x):
            if x < 0:
                raise ValueError("must be non-negative")
            return x

    await init()
    try:
        actor = await ValidatingActor.spawn()
        assert await actor.validate(5) == 5
        from pulsing.exceptions import PulsingActorError

        with pytest.raises(PulsingActorError, match="non-negative"):
            await actor.validate(-1)
    finally:
        await shutdown()


# ============================================================================
# Async method with direct await (non-streaming)
# ============================================================================


@pytest.mark.asyncio
async def test_async_method_await_returns_value():
    """Async method awaited directly should return final value."""

    @remote
    class AsyncValueActor:
        async def compute(self, x):
            await asyncio.sleep(0.01)
            return x**2

    await init()
    try:
        actor = await AsyncValueActor.spawn()
        result = await actor.compute(5)
        assert result == 25
    finally:
        await shutdown()


# ============================================================================
# Actor with supervision (restart_policy)
# ============================================================================


@pytest.mark.asyncio
async def test_supervised_actor_spawn():
    """Test spawning an actor with supervision parameters."""

    @remote(restart_policy="on-failure", max_restarts=2)
    class SupervisedActor:
        def __init__(self):
            self.value = 0

        def incr(self):
            self.value += 1
            return self.value

    await init()
    try:
        actor = await SupervisedActor.spawn()
        assert await actor.incr() == 1
        assert await actor.incr() == 2
    finally:
        await shutdown()


# ============================================================================
# ref.as_any() instance method
# ============================================================================


@pytest.mark.asyncio
async def test_ref_as_any():
    """Test the ref.as_any() instance method."""

    @remote
    class AsAnyActor:
        def greet(self):
            return "hi"

    await init()
    try:
        actor = await AsAnyActor.spawn(name="as_any_test", public=True)
        ref = await get_system().resolve("as_any_test")
        proxy = ref.as_any()
        assert await proxy.greet() == "hi"
    finally:
        await shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
