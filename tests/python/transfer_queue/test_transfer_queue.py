"""
Tests for the Pulsing Transfer Queue.

Covers:
- TransferBackend (incremental merge, get_data, clear, stats)
- StorageUnit @remote actor (put, get_data, clear, stats)
- AsyncTransferQueueClient (async_put, async_get, async_clear)
- TransferQueueClient (sync wrapper)
- Multi-bucket sharding
- Consumption tracking (task_name isolation)
- Concurrent writes
"""

import asyncio

import pytest

import pulsing as pul
from pulsing.transfer_queue.backend import TransferBackend
from pulsing.transfer_queue.client import AsyncTransferQueueClient
from pulsing.transfer_queue.storage import StorageUnit


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
def backend():
    """Create a TransferBackend for direct testing."""
    return TransferBackend(bucket_id=0, batch_size=10)


@pytest.fixture
async def client(actor_system):
    """Create an AsyncTransferQueueClient backed by a live actor system."""
    from pulsing.transfer_queue.manager import ensure_storage_managers

    await ensure_storage_managers(actor_system._inner)
    c = AsyncTransferQueueClient(
        partition_id="test", num_buckets=2, batch_size=10,
        system=actor_system._inner,
    )
    yield c
    await c.async_clear()


# ============================================================================
# TransferBackend Unit Tests
# ============================================================================


@pytest.mark.asyncio
async def test_backend_put_creates_sample(backend):
    """put() should create a new sample entry."""
    meta = await backend.put(0, {"prompt": "hello"})
    assert meta["sample_idx"] == 0
    assert "prompt" in meta["fields"]
    assert meta["status"] == "ok"


@pytest.mark.asyncio
async def test_backend_put_merges_fields(backend):
    """Successive put() calls merge fields into the same sample."""
    await backend.put(0, {"prompt": "hello"})
    meta = await backend.put(0, {"response": "world"})
    assert sorted(meta["fields"]) == ["prompt", "response"]


@pytest.mark.asyncio
async def test_backend_get_data_requires_all_fields(backend):
    """get_data only returns samples where all requested fields are present."""
    await backend.put(0, {"prompt": "hello"})
    # Only prompt written — requesting both should yield nothing
    rows = await backend.get_data(fields=["prompt", "response"], batch_size=10)
    assert rows == []

    # Now complete the sample
    await backend.put(0, {"response": "world"})
    rows = await backend.get_data(fields=["prompt", "response"], batch_size=10)
    assert len(rows) == 1
    assert rows[0]["prompt"] == "hello"
    assert rows[0]["response"] == "world"


@pytest.mark.asyncio
async def test_backend_get_data_respects_batch_size(backend):
    """get_data should return at most batch_size samples."""
    for i in range(5):
        await backend.put(i, {"a": i, "b": i})

    rows = await backend.get_data(fields=["a", "b"], batch_size=3)
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_backend_get_data_consumption_tracking(backend):
    """Consumed samples should not be returned again for the same task_name."""
    for i in range(3):
        await backend.put(i, {"x": i})

    batch1 = await backend.get_data(fields=["x"], batch_size=2, task_name="t1")
    assert len(batch1) == 2

    batch2 = await backend.get_data(fields=["x"], batch_size=2, task_name="t1")
    assert len(batch2) == 1  # only 1 left


@pytest.mark.asyncio
async def test_backend_query_cache_updates_incrementally(backend):
    """A cached query should see samples that become ready after the cache is built."""
    await backend.put(0, {"prompt": "hello"})

    rows = await backend.get_data(fields=["prompt", "response"], batch_size=10)
    assert rows == []

    await backend.put(0, {"response": "world"})

    rows = await backend.get_data(fields=["prompt", "response"], batch_size=10)
    assert rows == [{"prompt": "hello", "response": "world"}]


@pytest.mark.asyncio
async def test_backend_query_cache_tracks_repeated_reads_in_stats(backend):
    """Repeated reads of the same field-set should reuse the cached query state."""
    await backend.put(0, {"x": 1})

    await backend.get_data(fields=["x"], batch_size=1, task_name="t1")
    await backend.get_data(fields=["x"], batch_size=1, task_name="t2")

    stats = await backend.stats()
    assert stats["cached_queries"] >= 1
    assert stats["cached_query_hits"]["x"] >= 1
    assert stats["implementation"] == "indexed"


@pytest.mark.asyncio
async def test_backend_get_data_different_tasks_independent(backend):
    """Different task_names have independent consumption tracking."""
    for i in range(3):
        await backend.put(i, {"x": i})

    await backend.get_data(fields=["x"], batch_size=3, task_name="t1")
    rows = await backend.get_data(fields=["x"], batch_size=3, task_name="t2")
    assert len(rows) == 3  # t2 hasn't consumed anything


@pytest.mark.asyncio
async def test_backend_get_data(backend):
    """get_data returns correct sample content."""
    await backend.put(0, {"prompt": "hello", "response": "world"})
    await backend.put(1, {"prompt": "foo", "response": "bar"})

    rows = await backend.get_data(fields=["prompt", "response"], batch_size=2)
    assert len(rows) == 2
    assert rows[0]["prompt"] == "hello"
    assert rows[1]["response"] == "bar"


@pytest.mark.asyncio
async def test_backend_get_data_with_field_filter(backend):
    """get_data respects the fields filter."""
    await backend.put(0, {"prompt": "hello", "response": "world", "extra": 42})

    rows = await backend.get_data(fields=["prompt", "response"], batch_size=1)
    assert len(rows) == 1
    assert "extra" not in rows[0]
    assert rows[0]["prompt"] == "hello"


@pytest.mark.asyncio
async def test_backend_clear(backend):
    """clear() resets all state."""
    await backend.put(0, {"a": 1})
    await backend.clear()

    rows = await backend.get_data(fields=["a"], batch_size=10)
    assert rows == []


@pytest.mark.asyncio
async def test_backend_stats(backend):
    """stats() returns diagnostic info."""
    await backend.put(0, {"a": 1})
    await backend.put(1, {"a": 2})

    s = await backend.stats()
    assert s["bucket_id"] == 0
    assert s["sample_count"] == 2
    assert s["backend"] == "transfer_memory"


# ============================================================================
# StorageUnit Actor Tests
# ============================================================================


@pytest.fixture
async def storage_unit(actor_system):
    """Spawn a StorageUnit actor for testing."""
    proxy = await StorageUnit.spawn(
        bucket_id=0,
        batch_size=10,
        system=actor_system._inner,
        name="test_storage_unit_0",
        public=True,
    )
    yield proxy


@pytest.mark.asyncio
async def test_storage_unit_put(storage_unit):
    """StorageUnit.put merges data and returns meta."""
    meta = await storage_unit.put(sample_idx=0, data={"prompt": "hello"})
    assert meta["status"] == "ok"
    assert meta["sample_idx"] == 0


@pytest.mark.asyncio
async def test_storage_unit_get_data(storage_unit):
    """StorageUnit supports get_data directly."""
    await storage_unit.put(sample_idx=0, data={"a": 1, "b": 2})
    await storage_unit.put(sample_idx=1, data={"a": 3, "b": 4})

    rows = await storage_unit.get_data(fields=["a", "b"], batch_size=10)
    assert len(rows) == 2
    assert rows[0]["a"] == 1
    assert rows[1]["a"] == 3


@pytest.mark.asyncio
async def test_storage_unit_clear(storage_unit):
    """StorageUnit.clear resets backend."""
    await storage_unit.put(sample_idx=0, data={"x": 1})
    result = await storage_unit.clear()
    assert result["status"] == "ok"

    rows = await storage_unit.get_data(fields=["x"], batch_size=10)
    assert rows == []


@pytest.mark.asyncio
async def test_storage_unit_stats(storage_unit):
    """StorageUnit.stats returns backend diagnostics."""
    await storage_unit.put(sample_idx=0, data={"x": 1})
    s = await storage_unit.stats()
    assert s["sample_count"] == 1


# ============================================================================
# AsyncTransferQueueClient Tests
# ============================================================================


@pytest.mark.asyncio
async def test_client_put_and_get(client):
    """End-to-end: put incremental fields, then get complete samples."""
    await client.async_put(sample_idx=0, data={"prompt": "hello"})
    await client.async_put(sample_idx=0, data={"response": "world"})
    await client.async_put(sample_idx=1, data={"prompt": "foo"})
    await client.async_put(sample_idx=1, data={"response": "bar"})

    samples = await client.async_get(
        data_fields=["prompt", "response"], batch_size=10, task_name="train"
    )
    assert len(samples) == 2

    prompts = sorted(s["prompt"] for s in samples)
    responses = sorted(s["response"] for s in samples)
    assert prompts == ["foo", "hello"]
    assert responses == ["bar", "world"]


@pytest.mark.asyncio
async def test_client_get_incomplete_samples_excluded(client):
    """Samples missing required fields are not returned."""
    await client.async_put(sample_idx=0, data={"prompt": "hello"})
    # sample 0 only has prompt — response missing

    samples = await client.async_get(
        data_fields=["prompt", "response"], batch_size=10
    )
    assert len(samples) == 0


@pytest.mark.asyncio
async def test_client_get_respects_batch_size(client):
    """async_get returns at most batch_size samples."""
    for i in range(5):
        await client.async_put(sample_idx=i, data={"a": i, "b": i})

    samples = await client.async_get(data_fields=["a", "b"], batch_size=3)
    assert len(samples) == 3


@pytest.mark.asyncio
async def test_client_consumption_tracking(client):
    """Same task_name should not receive the same sample twice."""
    for i in range(3):
        await client.async_put(sample_idx=i, data={"x": i})

    batch1 = await client.async_get(
        data_fields=["x"], batch_size=2, task_name="consumer"
    )
    assert len(batch1) == 2

    batch2 = await client.async_get(
        data_fields=["x"], batch_size=2, task_name="consumer"
    )
    assert len(batch2) == 1

    batch3 = await client.async_get(
        data_fields=["x"], batch_size=2, task_name="consumer"
    )
    assert len(batch3) == 0


@pytest.mark.asyncio
async def test_client_clear(client):
    """async_clear resets all buckets."""
    for i in range(4):
        await client.async_put(sample_idx=i, data={"x": i})

    await client.async_clear()

    samples = await client.async_get(data_fields=["x"], batch_size=10)
    assert len(samples) == 0


@pytest.mark.asyncio
async def test_client_put_returns_meta(client):
    """async_put returns a BatchMeta-compatible dict."""
    meta = await client.async_put(sample_idx=42, data={"prompt": "hi"})
    assert meta["partition_id"] == "test"
    assert meta["sample_idx"] == 42
    assert "prompt" in meta["fields"]
    assert meta["status"] == "ok"


@pytest.mark.asyncio
async def test_client_multi_bucket_distribution(client):
    """Samples are sharded across buckets by sample_idx % num_buckets."""
    # client has num_buckets=2
    for i in range(6):
        await client.async_put(sample_idx=i, data={"v": i})

    # Even indices (0,2,4) go to bucket 0; odd (1,3,5) go to bucket 1
    assert 0 in client._bucket_refs
    assert 1 in client._bucket_refs


@pytest.mark.asyncio
async def test_client_concurrent_writes(client):
    """Concurrent async_put calls should not lose data."""
    num_samples = 20

    async def write(idx):
        await client.async_put(sample_idx=idx, data={"a": idx, "b": idx * 10})

    await asyncio.gather(*(write(i) for i in range(num_samples)))

    samples = await client.async_get(
        data_fields=["a", "b"], batch_size=num_samples
    )
    assert len(samples) == num_samples


# ============================================================================
# TransferQueueClient (Sync Wrapper) Tests
# ============================================================================


def _make_sync_env():
    """Helper: spin up a background event loop and return (loop, loop_thread, system, sync_client)."""
    import threading

    from pulsing.transfer_queue.client import AsyncTransferQueueClient, TransferQueueClient
    from pulsing.transfer_queue.manager import ensure_storage_managers

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    async def setup():
        system = await pul.actor_system()
        await ensure_storage_managers(system._inner)
        inner = AsyncTransferQueueClient(
            partition_id="sync_test", num_buckets=2, batch_size=10,
            system=system._inner,
        )
        return system, inner

    future = asyncio.run_coroutine_threadsafe(setup(), loop)
    system, inner = future.result(timeout=10)
    sync_client = TransferQueueClient(inner, loop)
    return loop, loop_thread, system, sync_client


def _teardown_sync_env(loop, loop_thread, system):
    """Helper: shutdown system and stop the background loop."""
    async def cleanup():
        await system.shutdown()

    asyncio.run_coroutine_threadsafe(cleanup(), loop).result(timeout=10)
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=5)


def test_sync_client_put_and_get():
    """Sync client: incremental put then get complete samples."""
    loop, loop_thread, system, client = _make_sync_env()
    try:
        client.put(sample_idx=0, data={"prompt": "hello"})
        client.put(sample_idx=0, data={"response": "world"})

        samples = client.get(
            data_fields=["prompt", "response"], batch_size=10, task_name="sync"
        )
        assert len(samples) == 1
        assert samples[0]["prompt"] == "hello"
        assert samples[0]["response"] == "world"
    finally:
        _teardown_sync_env(loop, loop_thread, system)


def test_sync_client_incomplete_excluded():
    """Sync client: incomplete samples are not returned."""
    loop, loop_thread, system, client = _make_sync_env()
    try:
        client.put(sample_idx=0, data={"prompt": "hello"})
        # response missing

        samples = client.get(
            data_fields=["prompt", "response"], batch_size=10
        )
        assert len(samples) == 0
    finally:
        _teardown_sync_env(loop, loop_thread, system)


def test_sync_client_batch_size():
    """Sync client: get respects batch_size."""
    loop, loop_thread, system, client = _make_sync_env()
    try:
        for i in range(5):
            client.put(sample_idx=i, data={"a": i, "b": i})

        samples = client.get(data_fields=["a", "b"], batch_size=3)
        assert len(samples) == 3
    finally:
        _teardown_sync_env(loop, loop_thread, system)


def test_sync_client_consumption_tracking():
    """Sync client: same task_name does not receive the same sample twice."""
    loop, loop_thread, system, client = _make_sync_env()
    try:
        for i in range(4):
            client.put(sample_idx=i, data={"x": i})

        batch1 = client.get(data_fields=["x"], batch_size=2, task_name="c1")
        assert len(batch1) == 2

        batch2 = client.get(data_fields=["x"], batch_size=2, task_name="c1")
        assert len(batch2) == 2

        batch3 = client.get(data_fields=["x"], batch_size=2, task_name="c1")
        assert len(batch3) == 0
    finally:
        _teardown_sync_env(loop, loop_thread, system)


def test_sync_client_multi_bucket():
    """Sync client: data is sharded across buckets."""
    loop, loop_thread, system, client = _make_sync_env()
    try:
        # num_buckets=2, so even/odd go to different buckets
        for i in range(6):
            client.put(sample_idx=i, data={"v": i})

        samples = client.get(data_fields=["v"], batch_size=10)
        assert len(samples) == 6
        values = sorted(s["v"] for s in samples)
        assert values == list(range(6))
    finally:
        _teardown_sync_env(loop, loop_thread, system)


def test_sync_client_clear():
    """Sync client: clear resets all data."""
    loop, loop_thread, system, client = _make_sync_env()
    try:
        for i in range(3):
            client.put(sample_idx=i, data={"x": i})

        client.clear()

        samples = client.get(data_fields=["x"], batch_size=10)
        assert len(samples) == 0
    finally:
        _teardown_sync_env(loop, loop_thread, system)


def test_sync_client_put_returns_meta():
    """Sync client: put returns a BatchMeta-compatible dict."""
    loop, loop_thread, system, client = _make_sync_env()
    try:
        meta = client.put(sample_idx=7, data={"prompt": "hi"})
        assert meta["partition_id"] == "sync_test"
        assert meta["sample_idx"] == 7
        assert "prompt" in meta["fields"]
        assert meta["status"] == "ok"
    finally:
        _teardown_sync_env(loop, loop_thread, system)


def test_sync_client_partition_id():
    """Sync client: partition_id property is accessible."""
    loop, loop_thread, system, client = _make_sync_env()
    try:
        assert client.partition_id == "sync_test"
    finally:
        _teardown_sync_env(loop, loop_thread, system)
