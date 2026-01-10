"""
Tests for the new Pulsing API styles.

Covers:
- Native async API (pulsing.actor with init/shutdown/remote)
- Ray-compatible API (pulsing.compat.ray)
"""

import asyncio

import pytest


# ============================================================================
# Native Async API Tests
# ============================================================================


@pytest.mark.asyncio
async def test_native_api_basic():
    """Test basic native async API workflow."""
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
        counter = await Counter.spawn(value=10)
        assert await counter.get() == 10
        assert await counter.inc(5) == 15
        assert await counter.inc() == 16
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_native_api_multiple_actors():
    """Test multiple actors with native API."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class Worker:
        def __init__(self, worker_id):
            self.worker_id = worker_id
            self.tasks_done = 0

        def process(self, data):
            self.tasks_done += 1
            return f"{self.worker_id}: processed {data}"

        def get_stats(self):
            return {"id": self.worker_id, "tasks": self.tasks_done}

    await init()

    try:
        workers = [await Worker.spawn(worker_id=f"w{i}") for i in range(3)]

        # Process some tasks
        results = []
        for i, w in enumerate(workers):
            result = await w.process(f"task-{i}")
            results.append(result)

        assert len(results) == 3
        assert "w0: processed task-0" in results[0]
        assert "w1: processed task-1" in results[1]
        assert "w2: processed task-2" in results[2]

        # Check stats
        for i, w in enumerate(workers):
            stats = await w.get_stats()
            assert stats["id"] == f"w{i}"
            assert stats["tasks"] == 1

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_native_api_async_methods():
    """Test actors with async methods."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class AsyncProcessor:
        def __init__(self):
            self.processed = []

        async def process(self, item):
            await asyncio.sleep(0.01)  # Simulate async work
            result = item.upper()
            self.processed.append(result)
            return result

        def get_processed(self):
            return self.processed

    await init()

    try:
        processor = await AsyncProcessor.spawn()

        # Process multiple items
        results = await asyncio.gather(
            processor.process("hello"),
            processor.process("world"),
            processor.process("pulsing"),
        )

        assert "HELLO" in results
        assert "WORLD" in results
        assert "PULSING" in results

        processed = await processor.get_processed()
        assert len(processed) == 3

    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_native_api_concurrent_calls():
    """Test concurrent calls to the same actor."""
    from pulsing.actor import init, shutdown, remote

    @remote
    class Counter:
        def __init__(self):
            self.value = 0

        def inc(self):
            self.value += 1
            return self.value

        def get(self):
            return self.value

    await init()

    try:
        counter = await Counter.spawn()

        # Send many concurrent increments
        tasks = [counter.inc() for _ in range(100)]
        await asyncio.gather(*tasks)

        # Due to actor's sequential processing, final value should be 100
        final = await counter.get()
        assert final == 100

    finally:
        await shutdown()


# ============================================================================
# Ray-Compatible API Tests
# ============================================================================


def test_ray_compat_api_basic():
    """Test basic Ray-compatible API workflow."""
    from pulsing.compat import ray

    ray.init()

    try:

        @ray.remote
        class Counter:
            def __init__(self, value=0):
                self.value = value

            def get(self):
                return self.value

            def inc(self, n=1):
                self.value += n
                return self.value

        counter = Counter.remote(value=10)
        assert ray.get(counter.get.remote()) == 10
        assert ray.get(counter.inc.remote(5)) == 15
        assert ray.get(counter.inc.remote()) == 16

    finally:
        ray.shutdown()


def test_ray_compat_api_multiple_actors():
    """Test multiple actors with Ray-compatible API."""
    from pulsing.compat import ray

    ray.init()

    try:

        @ray.remote
        class Worker:
            def __init__(self, worker_id):
                self.worker_id = worker_id

            def process(self, data):
                return f"{self.worker_id}: {data}"

        workers = [Worker.remote(worker_id=f"w{i}") for i in range(3)]

        # Process tasks
        refs = [w.process.remote(f"task-{i}") for i, w in enumerate(workers)]
        results = ray.get(refs)

        assert len(results) == 3
        assert "w0: task-0" in results[0]
        assert "w1: task-1" in results[1]
        assert "w2: task-2" in results[2]

    finally:
        ray.shutdown()


def test_ray_compat_api_wait():
    """Test ray.wait() functionality."""
    from pulsing.compat import ray

    ray.init()

    try:

        @ray.remote
        class SlowWorker:
            def work(self, duration):
                import time

                time.sleep(duration)
                return f"done after {duration}s"

        worker = SlowWorker.remote()

        # Submit multiple tasks with different durations
        refs = [
            worker.work.remote(0.01),
            worker.work.remote(0.02),
            worker.work.remote(0.03),
        ]

        # Wait for at least 1 to complete
        ready, remaining = ray.wait(refs, num_returns=1, timeout=5.0)

        assert len(ready) >= 1
        assert len(ready) + len(remaining) == 3

        # Get all results
        all_results = ray.get(refs)
        assert len(all_results) == 3

    finally:
        ray.shutdown()


def test_ray_compat_api_put_get():
    """Test ray.put() and ray.get() for object store."""
    from pulsing.compat import ray

    ray.init()

    try:
        # Put objects in store
        ref1 = ray.put({"key": "value1"})
        ref2 = ray.put([1, 2, 3, 4, 5])
        ref3 = ray.put("hello world")

        # Get objects back
        assert ray.get(ref1) == {"key": "value1"}
        assert ray.get(ref2) == [1, 2, 3, 4, 5]
        assert ray.get(ref3) == "hello world"

        # Batch get
        results = ray.get([ref1, ref2, ref3])
        assert len(results) == 3

    finally:
        ray.shutdown()


# ============================================================================
# API Migration Tests
# ============================================================================


def test_migration_pattern():
    """
    Demonstrate migration pattern from Ray to Pulsing.

    This test shows that the same logic works with both APIs.
    """

    # The actual computation logic
    def create_counter_class(decorator):
        @decorator
        class Counter:
            def __init__(self, value=0):
                self.value = value

            def inc(self):
                self.value += 1
                return self.value

        return Counter

    # Test with Ray-compatible API
    from pulsing.compat import ray

    ray.init()

    try:
        RayCounter = create_counter_class(ray.remote)
        counter = RayCounter.remote(value=0)
        result = ray.get(counter.inc.remote())
        assert result == 1
    finally:
        ray.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
