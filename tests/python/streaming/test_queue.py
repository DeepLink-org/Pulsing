"""
Tests for the Pulsing Distributed Memory Queue.

Covers:
- Basic queue operations (put, get)
- Hash partitioning and bucketing
- Memory-based storage (default backend)
- Streaming and blocking
- Distributed consumption (rank/world_size)

Stress tests are in test_queue_stress.py.
"""

import asyncio
import hashlib
import shutil
import tempfile
import time
from types import SimpleNamespace

import pytest

import pulsing as pul
import pulsing.streaming.sync_queue as sync_queue_module
from pulsing.streaming import (
    BucketStorage,
    Queue,
    QueueReader,
    read_queue,
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
    path = tempfile.mkdtemp(prefix="queue_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
async def queue(actor_system, temp_storage_path):
    """Create a test queue."""
    q = Queue(
        system=actor_system,
        topic="test_queue",
        bucket_column="id",
        num_buckets=4,
        batch_size=10,
        storage_path=temp_storage_path,
    )
    yield q
    await q.flush()


# ============================================================================
# Basic Queue Tests
# ============================================================================


@pytest.mark.asyncio
async def test_queue_creation(actor_system, temp_storage_path):
    """Test queue creation."""
    q = Queue(
        system=actor_system,
        topic="test_queue",
        bucket_column="user_id",
        num_buckets=4,
        batch_size=100,
        storage_path=temp_storage_path,
    )
    assert q.topic == "test_queue"
    assert q.bucket_column == "user_id"
    assert q.num_buckets == 4


@pytest.mark.asyncio
async def test_put_single_record(queue):
    """Test writing a single record."""
    record = {"id": "test_1", "value": 100}
    result = await queue.put(record)

    assert result["status"] == "ok"
    assert "bucket_id" in result


@pytest.mark.asyncio
async def test_put_multiple_records(queue):
    """Test writing multiple records."""
    records = [{"id": f"test_{i}", "value": i} for i in range(10)]
    results = await queue.put(records)

    assert len(results) == 10
    for result in results:
        assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_put_missing_bucket_column(queue):
    """Test error when partition column is missing."""
    record = {"value": 100}  # Missing 'id' column

    with pytest.raises(ValueError, match="Missing partition column"):
        await queue.put(record)


@pytest.mark.asyncio
async def test_get_records(queue):
    """Test reading records."""
    # Write records
    for i in range(20):
        await queue.put({"id": f"test_{i}", "value": i})

    # Read records (should get from memory buffer)
    records = await queue.get(limit=20)

    assert len(records) > 0
    assert len(records) <= 20


@pytest.mark.asyncio
async def test_get_from_specific_bucket(queue):
    """Test reading from a specific bucket."""
    # Write records
    for i in range(20):
        await queue.put({"id": f"test_{i}", "value": i})

    # Read from bucket 0
    records = await queue.get(bucket_id=0, limit=100)

    # All records should hash to bucket 0
    for record in records:
        bucket_id = queue.get_bucket_id(record["id"])
        assert bucket_id == 0


@pytest.mark.asyncio
async def test_flush_with_memory_backend(queue, temp_storage_path):
    """Test flush with memory backend (no-op but should not error)."""
    # Write records
    for i in range(5):
        await queue.put({"id": f"test_{i}", "value": i})

    # Flush should work (no-op for memory backend)
    await queue.flush()

    # Data should still be readable
    records = await queue.get(limit=10)
    assert len(records) == 5


@pytest.mark.asyncio
async def test_stats(queue):
    """Test queue statistics."""
    # Write records
    for i in range(15):
        await queue.put({"id": f"test_{i}", "value": i})

    stats = await queue.stats()

    assert stats["topic"] == "test_queue"
    assert stats["num_buckets"] == 4
    assert "buckets" in stats


# ============================================================================
# Hash Partitioning Tests
# ============================================================================


@pytest.mark.asyncio
async def test_hash_partitioning(queue):
    """Test that records are distributed across buckets."""
    bucket_counts = {i: 0 for i in range(queue.num_buckets)}

    # Write many records with different IDs
    for i in range(100):
        result = await queue.put({"id": f"user_{i}", "value": i})
        bucket_counts[result["bucket_id"]] += 1

    # All buckets should have some records (probabilistic)
    non_empty_buckets = sum(1 for count in bucket_counts.values() if count > 0)
    assert non_empty_buckets >= 2  # At least 2 buckets should have data


@pytest.mark.asyncio
async def test_same_key_same_bucket(queue):
    """Test that same key always goes to same bucket."""
    key = "consistent_key"

    results = []
    for i in range(10):
        result = await queue.put({"id": key, "value": i})
        results.append(result["bucket_id"])

    # All should go to the same bucket
    assert len(set(results)) == 1


@pytest.mark.asyncio
async def test_get_bucket_id(queue):
    """Test bucket ID calculation."""
    # Manual verification
    value = "test_value"
    expected = int(hashlib.md5(value.encode()).hexdigest(), 16) % queue.num_buckets
    actual = queue.get_bucket_id(value)
    assert actual == expected


# ============================================================================
# Data Visibility Tests
# ============================================================================


@pytest.mark.asyncio
async def test_immediate_visibility(queue):
    """Test that data is immediately visible after write (before flush)."""
    # Write a record
    await queue.put({"id": "immediate_test", "value": 42})

    # Read immediately (no flush)
    records = await queue.get(limit=10)

    # Should find the record in memory buffer
    found = any(r.get("id") == "immediate_test" for r in records)
    assert found, "Record should be visible immediately after write"


@pytest.mark.asyncio
async def test_combined_writes(queue):
    """Test that multiple writes are all visible."""
    # Write first batch
    for i in range(15):
        await queue.put({"id": f"batch1_{i}", "value": i})
    await queue.flush()  # No-op for memory, but should work

    # Write second batch
    for i in range(5):
        await queue.put({"id": f"batch2_{i}", "value": i + 100})

    # Read all
    records = await queue.get(limit=100)

    # Should have both batches
    batch1_count = sum(1 for r in records if r.get("id", "").startswith("batch1_"))
    batch2_count = sum(1 for r in records if r.get("id", "").startswith("batch2_"))

    assert batch1_count > 0, "Should have batch1 records"
    assert batch2_count > 0, "Should have batch2 records"


# ============================================================================
# Writer/Reader API Tests
# ============================================================================


@pytest.mark.asyncio
async def test_write_queue_api(actor_system, temp_storage_path):
    """Test write_queue function."""
    writer = await write_queue(
        actor_system,
        topic="writer_test",
        bucket_column="id",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    assert isinstance(writer, Queue)

    # Write data
    result = await writer.put({"id": "test", "value": 1})
    assert result["status"] == "ok"

    await writer.flush()


@pytest.mark.asyncio
async def test_read_queue_api(actor_system, temp_storage_path):
    """Test read_queue function."""
    # First create queue with writer
    writer = await write_queue(
        actor_system,
        topic="reader_test",
        bucket_column="id",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    for i in range(10):
        await writer.put({"id": f"test_{i}", "value": i})
    await writer.flush()

    # Open reader
    reader = await read_queue(
        actor_system,
        topic="reader_test",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    assert isinstance(reader, QueueReader)

    # Read data
    records = await reader.get(limit=10)
    assert len(records) > 0


@pytest.mark.asyncio
async def test_reader_offset_management(actor_system, temp_storage_path):
    """Test reader offset auto-increment."""
    # Create and populate queue
    writer = await write_queue(
        actor_system,
        topic="offset_test",
        bucket_column="id",
        num_buckets=2,
        storage_path=temp_storage_path,
    )

    for i in range(20):
        await writer.put({"id": f"test_{i}", "value": i})
    await writer.flush()

    # Create reader
    reader = await read_queue(
        actor_system,
        topic="offset_test",
        num_buckets=2,
        storage_path=temp_storage_path,
    )

    # Read first batch
    records1 = await reader.get(limit=5)

    # Read second batch (should not overlap)
    await reader.get(limit=5)

    # Reset and read again
    reader.reset()
    records3 = await reader.get(limit=5)

    # First and third should have same records
    ids1 = set(r["id"] for r in records1)
    ids3 = set(r["id"] for r in records3)
    assert ids1 == ids3, "After reset, should read from beginning"


# ============================================================================
# Distributed Consumption Tests
# ============================================================================


@pytest.mark.asyncio
async def test_distributed_consumption_rank_assignment(actor_system, temp_storage_path):
    """Test bucket assignment with rank/world_size."""
    # Create queue
    writer = await write_queue(
        actor_system,
        topic="distributed_test",
        bucket_column="id",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    for i in range(40):
        await writer.put({"id": f"test_{i}", "value": i})
    await writer.flush()

    # Create two readers with different ranks
    reader0 = await read_queue(
        actor_system,
        topic="distributed_test",
        rank=0,
        world_size=2,
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    reader1 = await read_queue(
        actor_system,
        topic="distributed_test",
        rank=1,
        world_size=2,
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    # Verify bucket assignments
    assert reader0.bucket_ids == [0, 2]
    assert reader1.bucket_ids == [1, 3]


@pytest.mark.asyncio
async def test_distributed_consumption_no_overlap(actor_system, temp_storage_path):
    """Test that distributed readers don't read overlapping data."""
    # Create queue with specific bucket count
    writer = await write_queue(
        actor_system,
        topic="no_overlap_test",
        bucket_column="id",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    # Write data
    for i in range(100):
        await writer.put({"id": f"record_{i}", "value": i})
    await writer.flush()

    # Create two readers
    reader0 = await read_queue(
        actor_system,
        topic="no_overlap_test",
        rank=0,
        world_size=2,
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    reader1 = await read_queue(
        actor_system,
        topic="no_overlap_test",
        rank=1,
        world_size=2,
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    # Read all data from both readers
    records0 = await reader0.get(limit=100)
    records1 = await reader1.get(limit=100)

    # No overlap
    ids0 = set(r["id"] for r in records0)
    ids1 = set(r["id"] for r in records1)

    overlap = ids0 & ids1
    assert len(overlap) == 0, f"Readers should not have overlapping data: {overlap}"


@pytest.mark.asyncio
async def test_explicit_bucket_ids(actor_system, temp_storage_path):
    """Test reading from explicit bucket IDs."""
    writer = await write_queue(
        actor_system,
        topic="explicit_bucket_test",
        bucket_column="id",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    for i in range(40):
        await writer.put({"id": f"test_{i}", "value": i})
    await writer.flush()

    # Read only from bucket 0
    reader = await read_queue(
        actor_system,
        topic="explicit_bucket_test",
        bucket_ids=[0],
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    records = await reader.get(limit=100)

    # All records should be from bucket 0
    for record in records:
        bucket_id = writer.get_bucket_id(record["id"])
        assert bucket_id == 0


# ============================================================================
# Blocking/Wait Tests
# ============================================================================


@pytest.mark.asyncio
async def test_blocking_read_with_timeout(actor_system, temp_storage_path):
    """Test blocking read times out correctly."""
    writer = await write_queue(
        actor_system,
        topic="blocking_test",
        bucket_column="id",
        num_buckets=1,
        storage_path=temp_storage_path,
    )

    # Write some data so bucket exists
    await writer.put({"id": "seed", "value": 0})

    reader = await read_queue(
        actor_system,
        topic="blocking_test",
        num_buckets=1,
        storage_path=temp_storage_path,
    )

    # Read existing data first
    await reader.get(limit=10)

    # Now blocking read should timeout
    start = time.time()
    records = await reader.get(limit=10, wait=True, timeout=0.5)
    elapsed = time.time() - start

    assert elapsed >= 0.4  # Should have waited
    assert len(records) == 0  # No new data


@pytest.mark.asyncio
async def test_blocking_read_wakes_on_data(actor_system, temp_storage_path):
    """Test blocking read wakes up when new data arrives."""
    writer = await write_queue(
        actor_system,
        topic="wake_test",
        bucket_column="id",
        num_buckets=1,
        batch_size=100,
        storage_path=temp_storage_path,
    )

    # Seed the queue
    await writer.put({"id": "seed", "value": 0})

    reader = await read_queue(
        actor_system,
        topic="wake_test",
        num_buckets=1,
        storage_path=temp_storage_path,
    )

    # Read seed data
    await reader.get(limit=10)

    # Write data first, then read with wait
    await writer.put({"id": "new_data", "value": 42})

    # Reset reader to read from beginning
    reader.reset()

    # Should get all data including new data
    records = await reader.get(limit=20, wait=False)

    assert len(records) >= 2  # seed + new_data
    ids = [r.get("id") for r in records]
    assert "seed" in ids
    assert "new_data" in ids


# ============================================================================
# BucketStorage Direct Tests
# ============================================================================


@pytest.mark.asyncio
async def test_bucket_storage_direct(actor_system, temp_storage_path):
    """Test BucketStorage actor directly with memory backend via proxy."""
    bucket = await BucketStorage.spawn(
        bucket_id=0,
        storage_path=f"{temp_storage_path}/direct_bucket",
        batch_size=5,
        backend="memory",
        system=actor_system,
        name="test_bucket",
    )

    # Put records via proxy method
    for i in range(10):
        result = await bucket.put({"id": f"test_{i}", "value": i})
        assert result["status"] == "ok"

    # Get stats via proxy method
    stats = await bucket.stats()

    assert stats["bucket_id"] == 0
    assert stats["total_count"] == 10
    assert stats["backend"] == "memory"

    # Flush (no-op for memory backend)
    await bucket.flush()

    # Data should still be there
    stats = await bucket.stats()
    assert stats["total_count"] == 10


@pytest.mark.asyncio
async def test_bucket_storage_get(actor_system, temp_storage_path):
    """Test BucketStorage get method via proxy."""
    bucket = await BucketStorage.spawn(
        bucket_id=0,
        storage_path=f"{temp_storage_path}/get_bucket",
        batch_size=5,
        backend="memory",
        system=actor_system,
        name="test_bucket_get",
    )

    # Put records
    for i in range(10):
        await bucket.put({"id": f"test_{i}", "value": i})

    # Get records via proxy
    records = await bucket.get(limit=10, offset=0)
    assert len(records) == 10

    # Get with limit
    records = await bucket.get(limit=5)
    assert len(records) == 5


@pytest.mark.asyncio
async def test_bucket_storage_put_batch(actor_system, temp_storage_path):
    """Test BucketStorage put_batch method via proxy."""
    bucket = await BucketStorage.spawn(
        bucket_id=0,
        storage_path=f"{temp_storage_path}/batch_bucket",
        batch_size=100,
        backend="memory",
        system=actor_system,
        name="test_bucket_batch",
    )

    # Put batch of records
    records = [{"id": f"batch_{i}", "value": i * 10} for i in range(20)]
    result = await bucket.put_batch(records)

    assert result["status"] == "ok"
    assert result["count"] == 20

    # Verify via stats
    stats = await bucket.stats()
    assert stats["total_count"] == 20


# ============================================================================
# Sync Queue Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sync_queue_requires_global_runtime(queue):
    """Standalone actor_system() queues should not expose sync wrappers."""
    with pytest.raises(RuntimeError, match="require the global Pulsing runtime"):
        queue.sync()


@pytest.mark.asyncio
async def test_sync_writer_reader_use_global_bridge(temp_storage_path):
    """Sync queue wrappers should work on the global Pulsing bridge."""
    await pul.init()

    try:
        writer = await write_queue(
            pul.get_system(),
            "sync_global",
            bucket_column="id",
            num_buckets=2,
            batch_size=10,
            storage_path=temp_storage_path,
        )
        reader = await read_queue(
            pul.get_system(),
            "sync_global",
            num_buckets=2,
            storage_path=temp_storage_path,
        )

        sync_writer = writer.sync()
        sync_reader = reader.sync()

        for i in range(10):
            result = await asyncio.to_thread(
                sync_writer.put, {"id": f"item_{i}", "data": f"value_{i}"}
            )
            assert result["status"] == "ok"
        await asyncio.to_thread(sync_writer.flush)

        records = await asyncio.to_thread(sync_reader.get, 20)
        assert len(records) == 10
        ids = {r["id"] for r in records}
        assert ids == {f"item_{i}" for i in range(10)}
    finally:
        if pul.is_initialized():
            await pul.shutdown()


@pytest.mark.asyncio
async def test_sync_reader_offset_uses_global_bridge(temp_storage_path):
    """SyncQueueReader offset helpers should work with the global bridge."""
    await pul.init()

    try:
        writer = await write_queue(
            pul.get_system(),
            "offset_test",
            bucket_column="id",
            num_buckets=1,
            batch_size=100,
            storage_path=temp_storage_path,
        )
        reader = await read_queue(
            pul.get_system(),
            "offset_test",
            num_buckets=1,
            storage_path=temp_storage_path,
        )

        sync_writer = writer.sync()
        sync_reader = reader.sync()

        for i in range(10):
            result = await asyncio.to_thread(
                sync_writer.put, {"id": "same_key", "seq": i}
            )
            assert result["status"] == "ok"
        await asyncio.to_thread(sync_writer.flush)

        records1 = await asyncio.to_thread(sync_reader.get, 5)
        assert len(records1) == 5

        records2 = await asyncio.to_thread(sync_reader.get, 5)
        assert len(records2) == 5

        sync_reader.reset()
        all_records = await asyncio.to_thread(sync_reader.get, 20)
        assert len(all_records) == 10
    finally:
        if pul.is_initialized():
            await pul.shutdown()


def test_sync_queue_get_and_stats_use_bridge(monkeypatch):
    class FakeBucket:
        def __init__(self, bucket_id: int):
            self.bucket_id = bucket_id

        async def stats(self):
            return {"bucket_id": self.bucket_id, "count": self.bucket_id + 1}

    class FakeQueueImpl:
        def __init__(self):
            self.get_calls = []

        async def get(self, bucket_id, limit, offset, wait, timeout):
            self.get_calls.append((bucket_id, limit, offset, wait, timeout))
            return [{"bucket_id": bucket_id, "limit": limit, "offset": offset}]

        async def _ensure_bucket(self, bucket_id):
            return FakeBucket(bucket_id)

    fake_queue = SimpleNamespace(
        system=object(),
        topic="sync_unit",
        bucket_column="id",
        num_buckets=2,
        batch_size=10,
        storage_path="/tmp/sync_queue_unit",
        backend="memory",
        backend_options={},
    )
    impl = FakeQueueImpl()

    monkeypatch.setattr(
        sync_queue_module, "_ensure_global_runtime", lambda system: None
    )
    monkeypatch.setattr(
        sync_queue_module, "run_sync", lambda awaitable: asyncio.run(awaitable)
    )
    monkeypatch.setattr(
        sync_queue_module.SyncQueue,
        "_make_queue",
        lambda self: impl,
    )

    sync_queue = sync_queue_module.SyncQueue(fake_queue)

    rows = sync_queue.get(bucket_id=1, limit=2, offset=3, wait=True, timeout=0.4)
    stats = sync_queue.stats()

    assert rows == [{"bucket_id": 1, "limit": 2, "offset": 3}]
    assert impl.get_calls == [(1, 2, 3, True, 0.4)]
    assert stats == {
        "topic": "sync_unit",
        "bucket_column": "id",
        "num_buckets": 2,
        "buckets": {
            0: {"bucket_id": 0, "count": 1},
            1: {"bucket_id": 1, "count": 2},
        },
    }


def test_sync_queue_reader_set_offset_updates_all_buckets(monkeypatch):
    fake_queue = SimpleNamespace(
        system=object(),
        topic="sync_unit",
        bucket_column="id",
        num_buckets=3,
        batch_size=10,
        storage_path="/tmp/sync_queue_unit",
        backend="memory",
        backend_options={},
    )
    fake_reader = SimpleNamespace(queue=fake_queue, bucket_ids=None, _offsets={})

    monkeypatch.setattr(
        sync_queue_module, "_ensure_global_runtime", lambda system: None
    )

    reader = sync_queue_module.SyncQueueReader(fake_reader)
    reader.set_offset(7)

    assert reader._offsets == {0: 7, 1: 7, 2: 7}


def test_sync_queue_rejects_non_global_system(monkeypatch):
    local_system = object()
    global_system = object()
    fake_queue = SimpleNamespace(
        system=local_system,
        topic="sync_unit",
        bucket_column="id",
        num_buckets=1,
        batch_size=10,
        storage_path="/tmp/sync_queue_unit",
        backend="memory",
        backend_options={},
    )

    monkeypatch.setattr(pul, "is_initialized", lambda: True)
    monkeypatch.setattr(pul, "get_system", lambda: global_system)
    monkeypatch.setattr(sync_queue_module, "get_loop", lambda: object())

    with pytest.raises(RuntimeError, match="require the global Pulsing runtime"):
        sync_queue_module.SyncQueue(fake_queue)


def test_sync_queue_rejects_missing_dispatch_loop(monkeypatch):
    system = object()
    fake_queue = SimpleNamespace(
        system=system,
        topic="sync_unit",
        bucket_column="id",
        num_buckets=1,
        batch_size=10,
        storage_path="/tmp/sync_queue_unit",
        backend="memory",
        backend_options={},
    )

    monkeypatch.setattr(pul, "is_initialized", lambda: True)
    monkeypatch.setattr(pul, "get_system", lambda: system)
    monkeypatch.setattr(sync_queue_module, "get_loop", lambda: None)

    with pytest.raises(RuntimeError, match="require the global Pulsing runtime"):
        sync_queue_module.SyncQueue(fake_queue)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
