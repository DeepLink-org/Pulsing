"""
Tests for @remote decorator features.

Covers:
- resolve() and proxy() methods
- sync vs async method handling
- streaming responses for async methods
- _AsyncMethodResult await and async iteration
"""

import asyncio

import pytest


# ============================================================================
# resolve() and proxy() Tests
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_named_actor():
    """Test resolving a named actor via Class.resolve()."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class Counter:
        def __init__(self, value=0):
            self.value = value

        def get(self):
            return self.value

        def inc(self, n=1):
            self.value += n
            return self.value

    await init()

    try:
        # Create named actor
        original = await Counter.spawn(name="test_counter", value=10)
        assert await original.get() == 10

        # Resolve by name
        resolved = await Counter.resolve("test_counter")
        assert await resolved.get() == 10

        # Modify via resolved reference
        assert await resolved.inc(5) == 15

        # Verify change via original
        assert await original.get() == 15

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_proxy_from_actor_ref():
    """Test creating proxy from ActorRef via Class.proxy()."""
    from pulsing.actor import init, shutdown, remote, get_system

    @remote
    class Calculator:
        def __init__(self):
            self.result = 0

        def add(self, n):
            self.result += n
            return self.result

        def get_result(self):
            return self.result

    await init()

    try:
        # Create named actor
        calc = await Calculator.spawn(name="my_calc")
        await calc.add(10)

        # Get raw ActorRef and wrap with proxy
        system = get_system()
        raw_ref = await system.resolve_named("my_calc")

        proxy = Calculator.proxy(raw_ref)
        assert await proxy.get_result() == 10
        assert await proxy.add(5) == 15

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_proxy_method_validation():
    """Test that proxy validates method names when methods list is provided."""
    from pulsing.actor import init, shutdown, remote, ActorProxy, get_system

    @remote
    class Service:
        def valid_method(self):
            return "ok"

    await init()

    try:
        service = await Service.spawn(name="my_service")

        # Access valid method should work
        result = await service.valid_method()
        assert result == "ok"

        # Access invalid method should raise AttributeError
        with pytest.raises(AttributeError):
            _ = service.invalid_method

        # Dynamic proxy (no method list) allows any method
        system = get_system()
        raw_ref = await system.resolve_named("my_service")
        dynamic_proxy = ActorProxy.from_ref(raw_ref)

        # This creates the method caller but will fail on actual call
        caller = dynamic_proxy.any_method_name
        assert caller is not None

    finally:
        await shutdown()


# ============================================================================
# Sync vs Async Method Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sync_method_normal_response():
    """Test that sync methods return normal responses."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class SyncService:
        def __init__(self):
            self.data = []

        def add(self, item):
            self.data.append(item)
            return len(self.data)

        def get_all(self):
            return self.data

    await init()

    try:
        service = await SyncService.spawn()

        # Sync methods should work normally
        assert await service.add("a") == 1
        assert await service.add("b") == 2
        assert await service.get_all() == ["a", "b"]

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_async_method_streaming_await():
    """Test that async methods support await for final result."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class AsyncService:
        async def compute(self, x):
            await asyncio.sleep(0.01)
            return x * 2

    await init()

    try:
        service = await AsyncService.spawn()

        # Await should get final result
        result = await service.compute(5)
        assert result == 10

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_async_generator_streaming():
    """Test that async generators stream intermediate values."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class StreamingService:
        async def generate_numbers(self, count):
            for i in range(count):
                await asyncio.sleep(0.001)
                yield i
            # async generators cannot return values in Python

    await init()

    try:
        service = await StreamingService.spawn()

        # Collect streamed values - directly use async for, no await needed
        collected = []
        async for item in service.generate_numbers(5):
            collected.append(item)

        assert collected == [0, 1, 2, 3, 4]

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_async_method_does_not_block_actor():
    """Test that async methods don't block the actor from receiving new messages."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class NonBlockingService:
        def __init__(self):
            self.call_count = 0

        async def slow_operation(self, delay):
            await asyncio.sleep(delay)
            self.call_count += 1
            return f"done after {delay}s"

        def get_call_count(self):
            return self.call_count

    await init()

    try:
        service = await NonBlockingService.spawn()

        # Start a slow operation (wrap in async def to use create_task)
        async def run_slow():
            return await service.slow_operation(0.1)

        slow_task = asyncio.create_task(run_slow())

        # Immediately make another call - should not be blocked
        await asyncio.sleep(0.01)  # Small delay
        count = await service.get_call_count()
        # The slow operation hasn't finished yet, so count should be 0
        assert count == 0

        # Wait for slow operation to complete
        await slow_task
        await asyncio.sleep(0.01)  # Let the count update propagate

        count = await service.get_call_count()
        assert count == 1

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_mixed_sync_async_methods():
    """Test class with both sync and async methods."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class MixedService:
        def __init__(self):
            self.value = 0

        # Sync method
        def get_value(self):
            return self.value

        # Sync method
        def set_value(self, v):
            self.value = v
            return self.value

        # Async method
        async def async_increment(self, n=1):
            await asyncio.sleep(0.01)
            self.value += n
            return self.value

    await init()

    try:
        service = await MixedService.spawn()

        # Sync methods
        assert await service.get_value() == 0
        assert await service.set_value(10) == 10
        assert await service.get_value() == 10

        # Async method
        result = await service.async_increment(5)
        assert result == 15

        # Verify final state
        assert await service.get_value() == 15

    finally:
        await shutdown()


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sync_method_error_handling():
    """Test error handling in sync methods."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class ErrorService:
        def will_fail(self):
            raise ValueError("Intentional error")

    await init()

    try:
        service = await ErrorService.spawn()

        with pytest.raises(RuntimeError, match="Intentional error"):
            await service.will_fail()

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_async_method_error_handling():
    """Test error handling in async methods."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class AsyncErrorService:
        async def will_fail(self):
            await asyncio.sleep(0.01)
            raise ValueError("Async error")

    await init()

    try:
        service = await AsyncErrorService.spawn()

        with pytest.raises(RuntimeError, match="Async error"):
            await service.will_fail()

    finally:
        await shutdown()


# ============================================================================
# ActorProxy.from_ref Tests
# ============================================================================


@pytest.mark.asyncio
async def test_actor_proxy_from_ref_dynamic_mode():
    """Test ActorProxy.from_ref in dynamic mode (no method list)."""
    from pulsing.actor import init, shutdown, remote, ActorProxy, get_system

    @remote
    class DynamicService:
        def method_a(self):
            return "a"

        def method_b(self):
            return "b"

    await init()

    try:
        await DynamicService.spawn(name="dynamic_svc")

        system = get_system()
        raw_ref = await system.resolve_named("dynamic_svc")

        # Dynamic mode - any method name is allowed
        proxy = ActorProxy.from_ref(raw_ref)

        # These should work
        assert await proxy.method_a() == "a"
        assert await proxy.method_b() == "b"

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_actor_proxy_from_ref_with_async_methods():
    """Test ActorProxy.from_ref with explicit async_methods set."""
    from pulsing.actor import init, shutdown, remote, ActorProxy, get_system

    @remote
    class HybridService:
        def sync_method(self):
            return "sync"

        async def async_method(self):
            await asyncio.sleep(0.01)
            return "async"

    await init()

    try:
        await HybridService.spawn(name="hybrid_svc")

        system = get_system()
        raw_ref = await system.resolve_named("hybrid_svc")

        # Create proxy with async method info
        proxy = ActorProxy.from_ref(
            raw_ref,
            methods=["sync_method", "async_method"],
            async_methods={"async_method"},
        )

        # Sync method
        assert await proxy.sync_method() == "sync"

        # Async method
        result = await proxy.async_method()
        assert result == "async"

    finally:
        await shutdown()


# ============================================================================
# Module-level resolve() function Tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_resolve_function():
    """Test module-level resolve() function."""
    from pulsing.actor import init, shutdown, remote, resolve

    @remote
    class SimpleService:
        def ping(self):
            return "pong"

    await init()

    try:
        await SimpleService.spawn(name="simple_svc")

        # Use module-level resolve (dynamic mode)
        proxy = await resolve("simple_svc")
        assert await proxy.ping() == "pong"

    finally:
        await shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
