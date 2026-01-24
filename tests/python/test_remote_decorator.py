"""
Tests for @remote decorator advanced features.

NOTE: Basic @remote functionality (spawn, methods, streaming) is tested in:
- tests/python/apis/actor/test_actor_behavior.py
- tests/python/apis/ray_like/test_ray_like_api.py

This file covers advanced features not in the apis tests:
- ActorProxy.from_ref with method validation
- Error handling in methods
- Concurrent async method behavior
"""

import asyncio

import pytest


# ============================================================================
# ActorProxy Method Validation Tests
# ============================================================================


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
        service = await Service.spawn(name="my_service", public=True)

        # Access valid method should work
        result = await service.valid_method()
        assert result == "ok"

        # Access invalid method should raise AttributeError
        with pytest.raises(AttributeError):
            _ = service.invalid_method

        # Dynamic proxy (no method list) allows any method
        system = get_system()
        raw_ref = await system.resolve("my_service")
        dynamic_proxy = ActorProxy.from_ref(raw_ref)

        # This creates the method caller but will fail on actual call
        caller = dynamic_proxy.any_method_name
        assert caller is not None

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
        await DynamicService.spawn(name="dynamic_svc", public=True)

        system = get_system()
        raw_ref = await system.resolve("dynamic_svc")

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
        await HybridService.spawn(name="hybrid_svc", public=True)

        system = get_system()
        raw_ref = await system.resolve("hybrid_svc")

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
# Async Method Concurrency Tests
# ============================================================================


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

        # Start a slow operation
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
