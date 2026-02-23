"""
Tests for core/helpers.py and streaming utilities.

Focus on:
- Protocol unwrapping functions
- Response handling
- Stream message handling
"""

import asyncio

import pytest

import pulsing as pul
from pulsing.core import remote, init, shutdown


# ============================================================================
# Protocol unwrap tests
# ============================================================================


@pytest.mark.asyncio
async def test_unwrap_call():
    """Test wrap/unwrap call message."""
    from pulsing.core.remote import _wrap_call, _unwrap_call

    msg = _wrap_call("my_method", (1, 2, 3), {"key": "val"}, False)
    method, args, kwargs, is_async = _unwrap_call(msg)
    assert method == "my_method"
    assert args == (1, 2, 3)
    assert kwargs == {"key": "val"}
    assert is_async is False

    msg_async = _wrap_call("async_method", (), {"param": 42}, True)
    method, args, kwargs, is_async = _unwrap_call(msg_async)
    assert method == "async_method"
    assert args == ()
    assert kwargs == {"param": 42}
    assert is_async is True


@pytest.mark.asyncio
async def test_unwrap_response():
    """Test wrap/unwrap response message."""
    from pulsing.core.remote import _wrap_response, _unwrap_response

    resp = _wrap_response(result={"data": "success"})
    result, error = _unwrap_response(resp)
    assert error is None
    assert result == {"data": "success"}

    err = _wrap_response(error="something failed")
    result, error = _unwrap_response(err)
    assert result is None
    assert "something failed" in error


# ============================================================================
# Single value iterator
# ============================================================================


@pytest.mark.asyncio
async def test_single_value_iterator():
    """Test _SingleValueIterator yields one value then stops."""
    from pulsing.core.remote import _SingleValueIterator

    it = _SingleValueIterator("single_value")
    results = []
    async for v in it:
        results.append(v)
    assert results == ["single_value"]


# ============================================================================
# Delayed call proxy
# ============================================================================


@pytest.mark.asyncio
async def test_delayed_call_proxy_cancel():
    """Test that DelayedCallProxy tasks can be cancelled."""
    from pulsing.core.remote import _DelayedCallProxy

    await init()
    try:

        @remote
        class TestActor:
            def ping(self):
                return "pong"

        actor = await TestActor.spawn()
        # Get the ref through the ref property
        ref = actor.ref

        proxy = _DelayedCallProxy(ref, 0.1)
        # This returns a task that can be cancelled
        task = proxy.ping()
        task.cancel()

        # Wait a bit
        await asyncio.sleep(0.2)

    finally:
        await shutdown()


# ============================================================================
# Exception consuming
# ============================================================================


@pytest.mark.asyncio
async def test_consume_task_exception():
    """Test _consume_task_exception handles various exception types."""
    from pulsing.core.remote import _consume_task_exception

    async def raise_cancelled():
        raise asyncio.CancelledError()

    async def raise_runtime():
        raise RuntimeError("stream closed")

    async def raise_value():
        raise ValueError("bad value")

    # CancelledError should be silently consumed
    task = asyncio.create_task(raise_cancelled())
    try:
        await task
    except asyncio.CancelledError:
        pass
    _consume_task_exception(task)

    # RuntimeError should be logged but not raise
    task = asyncio.create_task(raise_runtime())
    try:
        await task
    except RuntimeError:
        pass
    _consume_task_exception(task)

    # ValueError should be logged
    task = asyncio.create_task(raise_value())
    try:
        await task
    except ValueError:
        pass
    _consume_task_exception(task)


# ============================================================================
# Error path tests
# ============================================================================


@pytest.mark.asyncio
async def test_actor_error_in_async_generator():
    """Test error handling in async generator."""

    @remote
    class FailingAsyncGenActor:
        async def failing_gen(self, fail_at):
            for i in range(10):
                if i == fail_at:
                    raise RuntimeError(f"Failed at {fail_at}")
                yield i

    await init()
    try:
        actor = await FailingAsyncGenActor.spawn()
        results = []
        with pytest.raises(Exception):
            async for v in actor.failing_gen(3):
                results.append(v)
        # May or may not have results depending on when exception raised
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_sync_generator_error():
    """Test error in sync generator."""

    @remote
    class FailingSyncGenActor:
        def failing_gen(self, fail_at):
            for i in range(10):
                if i == fail_at:
                    raise RuntimeError(f"Failed at {fail_at}")
                yield i

    await init()
    try:
        actor = await FailingSyncGenActor.spawn()
        result = await actor.failing_gen(3)
        items = []
        with pytest.raises(Exception):
            if hasattr(result, "__aiter__"):
                async for v in result:
                    items.append(v)
            elif hasattr(result, "__iter__"):
                for v in result:
                    items.append(v)
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_actor_error_in_on_start():
    """Test error in on_start - actor spawn should still succeed but method calls fail."""
    # Note: When on_start raises, the actor's mailbox may close
    # This test verifies the error path is exercised

    @remote
    class FailingOnStartActor:
        def __init__(self):
            self.started = False

        def on_start(self, actor_id):
            raise ValueError("on_start error")

        def ping(self):
            return "pong"

    await init()
    try:
        # Actor spawn should succeed
        actor = await FailingOnStartActor.spawn()
        # The on_start error may cause the actor to stop
        # This tests the error handling path
    finally:
        await shutdown()


# ============================================================================
# Actor lifecycle
# ============================================================================


@pytest.mark.asyncio
async def test_actor_lifecycle():
    """Test actor lifecycle - on_start callback."""

    lifecycle_events = []

    @remote
    class LifecycleActor:
        def __init__(self, events):
            self.events = events
            self.value = 0

        def on_start(self, actor_id):
            self.events.append(("on_start", str(actor_id)))

        def on_stop(self):
            self.events.append(("on_stop", None))

        def metadata(self):
            return {"type": "lifecycle"}

        def increment(self):
            self.value += 1
            return self.value

        def get_value(self):
            return self.value

    await init()
    try:
        actor = await LifecycleActor.spawn(lifecycle_events)

        # Ensure actor has started (on_start runs before first message is processed)
        _ = await actor.get_value()
        assert any(e[0] == "on_start" for e in lifecycle_events)

        # Use the actor
        assert await actor.increment() == 1
        assert await actor.increment() == 2

    finally:
        await shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
