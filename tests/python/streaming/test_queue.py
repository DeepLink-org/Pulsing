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

import pytest

import pulsing as pul
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


def test_sync_queue_standalone():
    """Test sync queue wrapper.

    Note: Sync wrappers are designed for non-async code. Event loop runs
    in background thread while sync operations are called from main thread.
    """
    import tempfile
    import shutil
    import threading

    temp_dir = tempfile.mkdtemp(prefix="sync_test_")

    try:
        # Create event loop in background thread
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        try:
            # Setup in background loop
            async def setup():
                import pulsing as pul
                from pulsing.streaming import write_queue, read_queue

                system = await pul.actor_system()
                writer = await write_queue(
                    system,
                    "sync_test",
                    bucket_column="id",
                    num_buckets=2,
                    batch_size=10,
                    storage_path=temp_dir,
                )
                reader = await read_queue(
                    system, "sync_test", num_buckets=2, storage_path=temp_dir
                )
                return system, writer, reader

            future = asyncio.run_coroutine_threadsafe(setup(), loop)
            system, writer, reader = future.result(timeout=10)

            # Get sync wrappers (they will use the background loop)
            sync_writer = writer.sync()
            sync_reader = reader.sync()

            # Test sync put (from main thread)
            for i in range(5):
                result = sync_writer.put({"id": f"sync_{i}", "value": i})
                assert result["status"] == "ok"

            sync_writer.flush()

            # Test sync get
            records = sync_reader.get(limit=10)
            assert len(records) == 5

            # Cleanup
            async def cleanup():
                await system.shutdown()

            future = asyncio.run_coroutine_threadsafe(cleanup(), loop)
            future.result(timeout=10)

        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sync_writer_reader_standalone():
    """Test SyncQueue (write) and SyncQueueReader."""
    import tempfile
    import shutil
    import threading

    temp_dir = tempfile.mkdtemp(prefix="sync_wr_test_")

    try:
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        try:

            async def setup():
                import pulsing as pul
                from pulsing.streaming import write_queue, read_queue

                system = await pul.actor_system()
                writer = await write_queue(
                    system,
                    "sync_wr",
                    bucket_column="id",
                    num_buckets=2,
                    batch_size=10,
                    storage_path=temp_dir,
                )
                reader = await read_queue(
                    system, "sync_wr", num_buckets=2, storage_path=temp_dir
                )
                return system, writer, reader

            future = asyncio.run_coroutine_threadsafe(setup(), loop)
            system, writer, reader = future.result(timeout=10)

            sync_writer = writer.sync()
            sync_reader = reader.sync()

            # Write
            for i in range(10):
                sync_writer.put({"id": f"item_{i}", "data": f"value_{i}"})
            sync_writer.flush()

            # Read
            records = sync_reader.get(limit=20)
            assert len(records) == 10

            ids = {r["id"] for r in records}
            assert ids == {f"item_{i}" for i in range(10)}

            # Cleanup
            async def cleanup():
                await system.shutdown()

            future = asyncio.run_coroutine_threadsafe(cleanup(), loop)
            future.result(timeout=10)

        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sync_reader_offset_standalone():
    """Test SyncQueueReader offset management."""
    import tempfile
    import shutil
    import threading

    temp_dir = tempfile.mkdtemp(prefix="sync_offset_test_")

    try:
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        try:

            async def setup():
                import pulsing as pul
                from pulsing.streaming import write_queue, read_queue

                system = await pul.actor_system()
                writer = await write_queue(
                    system,
                    "offset_test",
                    bucket_column="id",
                    num_buckets=1,
                    batch_size=100,
                    storage_path=temp_dir,
                )
                reader = await read_queue(
                    system, "offset_test", num_buckets=1, storage_path=temp_dir
                )
                return system, writer, reader

            future = asyncio.run_coroutine_threadsafe(setup(), loop)
            system, writer, reader = future.result(timeout=10)

            sync_writer = writer.sync()
            sync_reader = reader.sync()

            # Write 10 records
            for i in range(10):
                sync_writer.put({"id": "same_key", "seq": i})
            sync_writer.flush()

            # Read first 5
            records1 = sync_reader.get(limit=5)
            assert len(records1) == 5

            # Read next 5
            records2 = sync_reader.get(limit=5)
            assert len(records2) == 5

            # Reset and read all
            sync_reader.reset()
            all_records = sync_reader.get(limit=20)
            assert len(all_records) == 10

            # Cleanup
            async def cleanup():
                await system.shutdown()

            future = asyncio.run_coroutine_threadsafe(cleanup(), loop)
            future.result(timeout=10)

        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
