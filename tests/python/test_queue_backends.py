"""
Tests for the Pluggable Queue Backend System.

Covers:
- StorageBackend protocol
- MemoryBackend implementation (built-in)
- Backend registration mechanism
- Integration with write_queue/read_queue APIs
- Custom backend implementation

Note: LanceBackend tests are in persisting package.
"""

import asyncio
import shutil
import tempfile
from typing import Any, AsyncIterator

import pytest

import pulsing as pul
from pulsing.queue import (
    BucketStorage,
    MemoryBackend,
    Queue,
    StorageBackend,
    get_backend_class,
    list_backends,
    read_queue,
    register_backend,
    write_queue,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def actor_system():
    """Create a standalone actor system for testing."""
    system = await pul.actor_system()
    yield system
    await system.shutdown()


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for queue storage."""
    path = tempfile.mkdtemp(prefix="backend_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ============================================================================
# MemoryBackend Tests
# ============================================================================


class TestMemoryBackend:
    """Tests for MemoryBackend."""

    @pytest.mark.asyncio
    async def test_memory_backend_creation(self):
        """Test MemoryBackend can be created."""
        backend = MemoryBackend(bucket_id=0)
        assert backend.bucket_id == 0
        assert backend.total_count() == 0

    @pytest.mark.asyncio
    async def test_memory_backend_put_single(self):
        """Test single record put."""
        backend = MemoryBackend(bucket_id=0)
        await backend.put({"id": "test", "value": 42})
        assert backend.total_count() == 1

    @pytest.mark.asyncio
    async def test_memory_backend_put_batch(self):
        """Test batch record put."""
        backend = MemoryBackend(bucket_id=0)
        records = [{"id": f"test_{i}", "value": i} for i in range(10)]
        await backend.put_batch(records)
        assert backend.total_count() == 10

    @pytest.mark.asyncio
    async def test_memory_backend_get(self):
        """Test get records."""
        backend = MemoryBackend(bucket_id=0)
        for i in range(10):
            await backend.put({"id": f"test_{i}", "value": i})

        records = await backend.get(limit=5, offset=0)
        assert len(records) == 5
        assert records[0]["id"] == "test_0"

        records = await backend.get(limit=5, offset=5)
        assert len(records) == 5
        assert records[0]["id"] == "test_5"

    @pytest.mark.asyncio
    async def test_memory_backend_get_stream(self):
        """Test streaming get."""
        backend = MemoryBackend(bucket_id=0)
        for i in range(20):
            await backend.put({"id": f"test_{i}", "value": i})

        all_records = []
        async for batch in backend.get_stream(limit=20, offset=0):
            all_records.extend(batch)

        assert len(all_records) == 20

    @pytest.mark.asyncio
    async def test_memory_backend_get_stream_wait_timeout(self):
        """Test streaming get with wait and timeout."""
        backend = MemoryBackend(bucket_id=0)
        await backend.put({"id": "seed", "value": 0})

        # Read existing data
        _ = await backend.get(limit=10, offset=0)

        # Stream with wait should timeout
        import time

        start = time.time()
        records = []
        async for batch in backend.get_stream(
            limit=10, offset=1, wait=True, timeout=0.3
        ):
            records.extend(batch)
        elapsed = time.time() - start

        assert elapsed >= 0.2  # Should have waited
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_memory_backend_flush_is_noop(self):
        """Test flush is a no-op for memory backend."""
        backend = MemoryBackend(bucket_id=0)
        await backend.put({"id": "test", "value": 1})
        await backend.flush()  # Should not raise
        assert backend.total_count() == 1

    @pytest.mark.asyncio
    async def test_memory_backend_stats(self):
        """Test stats reporting."""
        backend = MemoryBackend(bucket_id=5)
        for i in range(10):
            await backend.put({"id": f"test_{i}", "value": i})

        stats = await backend.stats()
        assert stats["bucket_id"] == 5
        assert stats["buffer_size"] == 10
        assert stats["persisted_count"] == 0
        assert stats["total_count"] == 10
        assert stats["backend"] == "memory"

    @pytest.mark.asyncio
    async def test_memory_backend_condition_notify(self):
        """Test that put notifies waiting readers."""
        backend = MemoryBackend(bucket_id=0)
        await backend.put({"id": "seed", "value": 0})

        received = []
        read_started = asyncio.Event()

        async def reader():
            read_started.set()
            async for batch in backend.get_stream(
                limit=5, offset=1, wait=True, timeout=2.0
            ):
                received.extend(batch)
                if len(received) >= 2:
                    break

        reader_task = asyncio.create_task(reader())
        await read_started.wait()
        await asyncio.sleep(0.1)  # Let reader start waiting

        # Write new data - should wake up reader
        await backend.put({"id": "new1", "value": 1})
        await backend.put({"id": "new2", "value": 2})

        await asyncio.wait_for(reader_task, timeout=3.0)

        assert len(received) >= 2


# ============================================================================
# Backend Registration Tests
# ============================================================================


class TestBackendRegistration:
    """Tests for backend registration mechanism."""

    def test_list_builtin_backends(self):
        """Test listing built-in backends."""
        backends = list_backends()
        assert "memory" in backends

    def test_get_builtin_backend_by_name(self):
        """Test getting built-in backend by name."""
        memory_cls = get_backend_class("memory")
        assert memory_cls is MemoryBackend

    def test_get_backend_by_class(self):
        """Test getting backend by class directly."""
        result = get_backend_class(MemoryBackend)
        assert result is MemoryBackend

    def test_get_unknown_backend_raises(self):
        """Test getting unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_class("nonexistent")

    def test_register_custom_backend(self):
        """Test registering a custom backend."""

        class CustomBackend:
            def __init__(self, bucket_id: int, **kwargs):
                self.bucket_id = bucket_id

        register_backend("custom_test", CustomBackend)

        # Should appear in list
        assert "custom_test" in list_backends()

        # Should be retrievable
        result = get_backend_class("custom_test")
        assert result is CustomBackend

    def test_register_invalid_backend_raises(self):
        """Test registering non-class raises error."""
        with pytest.raises(TypeError, match="must be a class"):
            register_backend("invalid", "not_a_class")


# ============================================================================
# BucketStorage with Backend Tests
# ============================================================================


class TestBucketStorageWithBackend:
    """Tests for BucketStorage using different backends."""

    @pytest.mark.asyncio
    async def test_bucket_storage_with_memory_backend(
        self, actor_system, temp_storage_path
    ):
        """Test BucketStorage with memory backend."""
        from pulsing.actor import Message

        storage = BucketStorage(
            bucket_id=0,
            storage_path=f"{temp_storage_path}/bucket_memory",
            batch_size=10,
            backend="memory",
        )

        actor_ref = await actor_system.spawn(storage, name="bucket_memory_test")

        # Put records
        for i in range(5):
            response = await actor_ref.ask(
                Message.from_json("Put", {"record": {"id": f"test_{i}", "value": i}})
            )
            assert response.to_json().get("status") == "ok"

        # Get stats
        stats_response = await actor_ref.ask(Message.from_json("Stats", {}))
        stats = stats_response.to_json()

        assert stats["bucket_id"] == 0
        assert stats["total_count"] == 5
        assert stats["backend"] == "memory"

    @pytest.mark.asyncio
    async def test_bucket_storage_put_batch(self, actor_system, temp_storage_path):
        """Test BucketStorage PutBatch message."""
        from pulsing.actor import Message

        storage = BucketStorage(
            bucket_id=0,
            storage_path=f"{temp_storage_path}/bucket_batch",
            batch_size=100,
            backend="memory",
        )

        actor_ref = await actor_system.spawn(storage, name="bucket_batch_test")

        # Put batch
        records = [{"id": f"batch_{i}", "value": i} for i in range(10)]
        response = await actor_ref.ask(
            Message.from_json("PutBatch", {"records": records})
        )
        result = response.to_json()
        assert result.get("status") == "ok"
        assert result.get("count") == 10

        # Verify
        stats_response = await actor_ref.ask(Message.from_json("Stats", {}))
        stats = stats_response.to_json()
        assert stats["total_count"] == 10


# ============================================================================
# Queue API with Backend Tests
# ============================================================================


class TestQueueAPIWithBackend:
    """Tests for Queue/write_queue/read_queue with different backends."""

    @pytest.mark.asyncio
    async def test_write_queue_with_memory_backend(
        self, actor_system, temp_storage_path
    ):
        """Test write_queue with memory backend."""
        writer = await write_queue(
            actor_system,
            topic="memory_writer_test",
            bucket_column="id",
            num_buckets=2,
            storage_path=temp_storage_path,
            backend="memory",
        )

        # Write data
        for i in range(10):
            result = await writer.put({"id": f"test_{i}", "value": i})
            assert result["status"] == "ok"

        # Check stats
        stats = await writer.queue.stats()
        total = sum(b.get("total_count", 0) for b in stats["buckets"].values())
        assert total == 10

    @pytest.mark.asyncio
    async def test_read_queue_with_memory_backend(
        self, actor_system, temp_storage_path
    ):
        """Test read_queue with memory backend."""
        # Write data
        writer = await write_queue(
            actor_system,
            topic="memory_reader_test",
            bucket_column="id",
            num_buckets=2,
            storage_path=temp_storage_path,
            backend="memory",
        )

        for i in range(10):
            await writer.put({"id": f"test_{i}", "value": i})

        # Read data
        reader = await read_queue(
            actor_system,
            topic="memory_reader_test",
            num_buckets=2,
            storage_path=temp_storage_path,
            backend="memory",
        )

        records = await reader.get(limit=20)
        assert len(records) == 10

    @pytest.mark.asyncio
    async def test_queue_with_registered_backend(self, actor_system, temp_storage_path):
        """Test queue with registered custom backend."""
        # Register a custom backend
        register_backend("memory_registered", MemoryBackend)

        writer = await write_queue(
            actor_system,
            topic="registered_backend_test",
            bucket_column="id",
            num_buckets=2,
            storage_path=temp_storage_path,
            backend="memory_registered",
        )

        for i in range(5):
            result = await writer.put({"id": f"test_{i}", "value": i})
            assert result["status"] == "ok"

        stats = await writer.queue.stats()
        assert any(b.get("backend") == "memory" for b in stats["buckets"].values())

    @pytest.mark.asyncio
    async def test_default_backend_is_memory(self, actor_system, temp_storage_path):
        """Test that default backend is memory."""
        writer = await write_queue(
            actor_system,
            topic="default_backend_test",
            bucket_column="id",
            num_buckets=2,
            storage_path=temp_storage_path,
            # backend not specified, should use "memory"
        )

        await writer.put({"id": "test", "value": 1})

        stats = await writer.queue.stats()
        assert any(b.get("backend") == "memory" for b in stats["buckets"].values())


# ============================================================================
# Custom Backend Protocol Tests
# ============================================================================


class TestCustomBackendProtocol:
    """Tests for custom backend implementations."""

    @pytest.mark.asyncio
    async def test_custom_backend_implements_protocol(self):
        """Test that custom backend implementing protocol works."""

        class MinimalBackend:
            """Minimal backend implementation for testing."""

            def __init__(self, bucket_id: int, **kwargs):
                self.bucket_id = bucket_id
                self.data: list[dict] = []

            async def put(self, record: dict[str, Any]) -> None:
                self.data.append(record)

            async def put_batch(self, records: list[dict[str, Any]]) -> None:
                self.data.extend(records)

            async def get(self, limit: int, offset: int) -> list[dict[str, Any]]:
                return self.data[offset : offset + limit]

            async def get_stream(
                self,
                limit: int,
                offset: int,
                wait: bool = False,
                timeout: float | None = None,
            ) -> AsyncIterator[list[dict[str, Any]]]:
                yield self.data[offset : offset + limit]

            async def flush(self) -> None:
                pass

            async def stats(self) -> dict[str, Any]:
                return {
                    "bucket_id": self.bucket_id,
                    "total_count": len(self.data),
                    "backend": "minimal",
                }

            def total_count(self) -> int:
                return len(self.data)

        # Verify it satisfies protocol (duck typing)
        backend = MinimalBackend(bucket_id=0)
        assert isinstance(backend, StorageBackend)

        # Test basic operations
        await backend.put({"id": "test", "value": 1})
        assert backend.total_count() == 1

        records = await backend.get(limit=10, offset=0)
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_custom_backend_with_bucket_storage(
        self, actor_system, temp_storage_path
    ):
        """Test custom backend with BucketStorage actor."""
        from pulsing.actor import Message

        class TrackingBackend:
            """Backend that tracks all operations."""

            def __init__(self, bucket_id: int, **kwargs):
                self.bucket_id = bucket_id
                self.data: list[dict] = []
                self.operations: list[str] = []
                self._lock = asyncio.Lock()
                self._condition = asyncio.Condition(self._lock)

            async def put(self, record: dict[str, Any]) -> None:
                async with self._condition:
                    self.operations.append("put")
                    self.data.append(record)
                    self._condition.notify_all()

            async def put_batch(self, records: list[dict[str, Any]]) -> None:
                async with self._condition:
                    self.operations.append("put_batch")
                    self.data.extend(records)
                    self._condition.notify_all()

            async def get(self, limit: int, offset: int) -> list[dict[str, Any]]:
                self.operations.append("get")
                return self.data[offset : offset + limit]

            async def get_stream(
                self,
                limit: int,
                offset: int,
                wait: bool = False,
                timeout: float | None = None,
            ) -> AsyncIterator[list[dict[str, Any]]]:
                self.operations.append("get_stream")
                yield self.data[offset : offset + limit]

            async def flush(self) -> None:
                self.operations.append("flush")

            async def stats(self) -> dict[str, Any]:
                return {
                    "bucket_id": self.bucket_id,
                    "total_count": len(self.data),
                    "backend": "tracking",
                    "buffer_size": len(self.data),
                    "persisted_count": 0,
                    "operations": self.operations.copy(),
                }

            def total_count(self) -> int:
                return len(self.data)

        # Register and use
        register_backend("tracking", TrackingBackend)

        storage = BucketStorage(
            bucket_id=0,
            storage_path=f"{temp_storage_path}/tracking_test",
            batch_size=100,
            backend="tracking",
        )

        actor_ref = await actor_system.spawn(storage, name="tracking_bucket")

        # Perform operations
        await actor_ref.ask(Message.from_json("Put", {"record": {"id": "1"}}))
        await actor_ref.ask(Message.from_json("Put", {"record": {"id": "2"}}))
        await actor_ref.ask(Message.from_json("Get", {"limit": 10, "offset": 0}))
        await actor_ref.ask(Message.from_json("Flush", {}))

        # Check tracking
        stats_response = await actor_ref.ask(Message.from_json("Stats", {}))
        stats = stats_response.to_json()

        assert stats["backend"] == "tracking"
        assert "put" in stats["operations"]
        assert "get" in stats["operations"]
        assert "flush" in stats["operations"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestBackendIntegration:
    """Integration tests for backend system."""

    @pytest.mark.asyncio
    async def test_memory_backend_distributed_consumption(
        self, actor_system, temp_storage_path
    ):
        """Test distributed consumption with memory backend."""
        writer = await write_queue(
            actor_system,
            topic="dist_memory_test",
            bucket_column="id",
            num_buckets=4,
            storage_path=temp_storage_path,
            backend="memory",
        )

        # Write data
        for i in range(40):
            await writer.put({"id": f"record_{i}", "value": i})

        # Create distributed readers
        reader0 = await read_queue(
            actor_system,
            topic="dist_memory_test",
            rank=0,
            world_size=2,
            num_buckets=4,
            storage_path=temp_storage_path,
            backend="memory",
        )

        reader1 = await read_queue(
            actor_system,
            topic="dist_memory_test",
            rank=1,
            world_size=2,
            num_buckets=4,
            storage_path=temp_storage_path,
            backend="memory",
        )

        # Read and verify no overlap
        records0 = await reader0.get(limit=100)
        records1 = await reader1.get(limit=100)

        ids0 = set(r["id"] for r in records0)
        ids1 = set(r["id"] for r in records1)

        overlap = ids0 & ids1
        assert len(overlap) == 0, f"Should have no overlap: {overlap}"

    @pytest.mark.asyncio
    async def test_high_concurrency_with_memory_backend(
        self, actor_system, temp_storage_path
    ):
        """Stress test: high concurrency writes with memory backend."""
        writer = await write_queue(
            actor_system,
            topic="stress_memory",
            bucket_column="id",
            num_buckets=4,
            storage_path=temp_storage_path,
            backend="memory",
        )

        num_writers = 5
        records_per_writer = 50

        async def write_batch(writer_id: int):
            for i in range(records_per_writer):
                await writer.put(
                    {
                        "id": f"w{writer_id}_r{i}",
                        "writer_id": writer_id,
                        "seq": i,
                    }
                )

        # Concurrent writes
        tasks = [write_batch(i) for i in range(num_writers)]
        await asyncio.gather(*tasks)

        # Verify all written
        stats = await writer.queue.stats()
        total = sum(b.get("total_count", 0) for b in stats["buckets"].values())
        assert total == num_writers * records_per_writer


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
