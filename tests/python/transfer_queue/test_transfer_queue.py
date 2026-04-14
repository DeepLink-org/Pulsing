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
from pulsing._async_bridge import get_pulsing_loop, get_shared_loop, run_sync
from pulsing.transfer_queue.backend import TransferBackend
from pulsing.transfer_queue.client import AsyncTransferQueueClient
from pulsing.transfer_queue.manager import STORAGE_MANAGER_NAME, StorageManager
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


async def _resolve_local_storage_manager():
    system = pul.get_system()
    return await StorageManager.resolve(
        STORAGE_MANAGER_NAME,
        system=system,
        node_id=system.node_id.id,
    )


async def _wait_for_local_storage_manager(
    retries: int = 20,
    delay: float = 0.05,
):
    for attempt in range(retries):
        try:
            return await _resolve_local_storage_manager()
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)


def _resolve_local_storage_manager_sync():
    return run_sync(_resolve_local_storage_manager())


@pytest.fixture
async def client(actor_system):
    """Create an AsyncTransferQueueClient backed by a live actor system."""
    from pulsing.transfer_queue.manager import ensure_storage_managers

    await ensure_storage_managers(actor_system._inner)
    c = AsyncTransferQueueClient(
        partition_id="test",
        num_buckets=2,
        batch_size=10,
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
async def test_backend_get_data_sample_idxs_returns_requested_ready_samples(backend):
    """sample_idxs mode returns only ready requested samples in request order."""
    await backend.put(0, {"a": 0, "b": 0})
    await backend.put(1, {"a": 1})
    await backend.put(2, {"a": 2, "b": 2})

    rows = await backend.get_data(
        fields=["a", "b"],
        batch_size=10,
        sample_idxs=[2, 1, 0],
    )

    assert [row["sample_idx"] for row in rows] == [2, 0]
    assert rows[0]["a"] == 2
    assert rows[1]["b"] == 0


@pytest.mark.asyncio
async def test_backend_get_data_sample_idxs_skips_incomplete_until_ready(backend):
    """sample_idxs mode should expose rows only after all required fields arrive."""
    await backend.put(7, {"prompt": "hello"})

    rows = await backend.get_data(
        fields=["prompt", "response"],
        batch_size=10,
        sample_idxs=[7],
    )
    assert rows == []

    await backend.put(7, {"response": "world"})

    rows = await backend.get_data(
        fields=["prompt", "response"],
        batch_size=10,
        sample_idxs=[7],
    )
    assert rows == [{"sample_idx": 7, "prompt": "hello", "response": "world"}]


@pytest.mark.asyncio
async def test_backend_get_data_sample_idxs_honors_consumption_tracking(backend):
    """sample_idxs mode should still mark returned rows consumed per task."""
    await backend.put(4, {"x": 4})

    first = await backend.get_data(
        fields=["x"],
        batch_size=10,
        task_name="t1",
        sample_idxs=[4],
    )
    second = await backend.get_data(
        fields=["x"],
        batch_size=10,
        task_name="t1",
        sample_idxs=[4],
    )
    other_task = await backend.get_data(
        fields=["x"],
        batch_size=10,
        task_name="t2",
        sample_idxs=[4],
    )

    assert first == [{"sample_idx": 4, "x": 4}]
    assert second == []
    assert other_task == [{"sample_idx": 4, "x": 4}]


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
async def test_storage_unit_get_data_sample_idxs(storage_unit):
    """StorageUnit supports direct sample_idxs lookups."""
    await storage_unit.put(sample_idx=0, data={"a": 1, "b": 2})
    await storage_unit.put(sample_idx=1, data={"a": 3})
    await storage_unit.put(sample_idx=2, data={"a": 5, "b": 6})

    rows = await storage_unit.get_data(
        fields=["a", "b"],
        batch_size=10,
        sample_idxs=[2, 1, 0],
    )

    assert [row["sample_idx"] for row in rows] == [2, 0]


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

    samples = await client.async_get(data_fields=["prompt", "response"], batch_size=10)
    assert len(samples) == 0


@pytest.mark.asyncio
async def test_client_get_respects_batch_size(client):
    """async_get returns at most batch_size samples."""
    for i in range(5):
        await client.async_put(sample_idx=i, data={"a": i, "b": i})

    samples = await client.async_get(data_fields=["a", "b"], batch_size=3)
    assert len(samples) == 3


@pytest.mark.asyncio
async def test_client_get_waits_for_batch_completion(client):
    """async_get waits up to timeout for enough rows to satisfy batch_size."""
    import time

    await client.async_put(sample_idx=0, data={"x": 0})

    async def delayed_put():
        await asyncio.sleep(0.1)
        await client.async_put(sample_idx=1, data={"x": 1})

    producer = asyncio.create_task(delayed_put())
    start = time.monotonic()
    samples = await client.async_get(
        data_fields=["x"],
        batch_size=2,
        task_name="wait_batch",
        timeout=0.5,
    )
    elapsed = time.monotonic() - start
    await producer

    assert elapsed >= 0.09
    assert len(samples) == 2
    assert sorted(row["x"] for row in samples) == [0, 1]


@pytest.mark.asyncio
async def test_client_get_timeout_returns_partial_batch(client):
    """async_get returns a partial batch after waiting until timeout."""
    import time

    await client.async_put(sample_idx=0, data={"x": 0})

    start = time.monotonic()
    samples = await client.async_get(
        data_fields=["x"],
        batch_size=2,
        task_name="partial_batch",
        timeout=0.3,
    )
    elapsed = time.monotonic() - start

    assert elapsed >= 0.25
    assert samples == [{"x": 0}]


@pytest.mark.asyncio
async def test_client_get_sample_idxs_across_buckets_preserves_order(client):
    """sample_idxs mode should preserve request order across buckets."""
    await client.async_put(sample_idx=0, data={"value": "zero"})
    await client.async_put(sample_idx=3, data={"value": "three"})

    rows = await client.async_get(
        data_fields=["value"],
        sample_idxs=[3, 0],
        batch_size=2,
        task_name="ordered_lookup",
    )

    assert [row["sample_idx"] for row in rows] == [3, 0]
    assert [row["value"] for row in rows] == ["three", "zero"]


@pytest.mark.asyncio
async def test_client_get_sample_idxs_duplicate_idx_raises(client):
    """sample_idxs mode should reject duplicate sample_idx values."""
    with pytest.raises(ValueError, match="duplicate sample_idx"):
        await client.async_get(
            data_fields=["value"],
            sample_idxs=[3, 0, 3],
            task_name="duplicate_lookup",
        )


@pytest.mark.asyncio
async def test_client_get_sample_idxs_batch_size_mismatch_raises(client):
    """sample_idxs mode should reject batch_size mismatches."""
    with pytest.raises(ValueError, match="batch_size must equal len\\(sample_idxs\\)"):
        await client.async_get(
            data_fields=["value"],
            sample_idxs=[3, 0],
            batch_size=1,
            task_name="mismatch_lookup",
        )


@pytest.mark.asyncio
async def test_client_get_sample_idxs_empty_list_returns_empty(client):
    """sample_idxs=[] remains valid when batch_size is omitted or zero."""
    rows_without_batch = await client.async_get(
        data_fields=["value"],
        sample_idxs=[],
        task_name="empty_lookup",
    )
    rows_with_zero_batch = await client.async_get(
        data_fields=["value"],
        sample_idxs=[],
        batch_size=0,
        task_name="empty_lookup_zero",
    )

    assert rows_without_batch == []
    assert rows_with_zero_batch == []


@pytest.mark.asyncio
async def test_client_get_sample_idxs_empty_list_nonzero_batch_raises(client):
    """sample_idxs=[] should reject conflicting nonzero batch_size."""
    with pytest.raises(ValueError, match="batch_size must equal len\\(sample_idxs\\)"):
        await client.async_get(
            data_fields=["value"],
            sample_idxs=[],
            batch_size=1,
            task_name="empty_lookup_bad",
        )


@pytest.mark.asyncio
async def test_client_get_sample_idxs_waits_for_missing_samples(client):
    """sample_idxs mode waits for unresolved samples until timeout expires."""
    import time

    await client.async_put(sample_idx=0, data={"prompt": "zero"})
    await client.async_put(sample_idx=0, data={"response": "done"})
    await client.async_put(sample_idx=1, data={"prompt": "one"})

    async def delayed_put():
        await asyncio.sleep(0.1)
        await client.async_put(sample_idx=1, data={"response": "later"})

    producer = asyncio.create_task(delayed_put())
    start = time.monotonic()
    rows = await client.async_get(
        data_fields=["prompt", "response"],
        sample_idxs=[1, 0],
        batch_size=2,
        task_name="sample_wait",
        timeout=0.6,
    )
    elapsed = time.monotonic() - start
    await producer

    assert elapsed >= 0.09
    assert [row["sample_idx"] for row in rows] == [1, 0]
    assert rows[0]["response"] == "later"
    assert rows[1]["response"] == "done"


@pytest.mark.asyncio
async def test_client_get_sample_idxs_timeout_returns_ready_subset(client):
    """sample_idxs mode returns ready subset with sample_idx after timeout."""
    import time

    await client.async_put(sample_idx=0, data={"prompt": "zero"})
    await client.async_put(sample_idx=0, data={"response": "done"})
    await client.async_put(sample_idx=1, data={"prompt": "one"})

    start = time.monotonic()
    rows = await client.async_get(
        data_fields=["prompt", "response"],
        sample_idxs=[1, 0],
        batch_size=2,
        task_name="sample_timeout",
        timeout=0.3,
    )
    elapsed = time.monotonic() - start

    assert elapsed >= 0.25
    assert rows == [{"sample_idx": 0, "prompt": "zero", "response": "done"}]


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

    samples = await client.async_get(data_fields=["a", "b"], batch_size=num_samples)
    assert len(samples) == num_samples


# ============================================================================
# TransferQueueClient (Sync Wrapper) Tests
# ============================================================================


def _make_sync_client() -> object:
    """Create a sync transfer queue client via the supported public API."""
    if pul.is_initialized():
        pul.transfer_queue.shutdown()
    assert not pul.is_initialized()
    return pul.transfer_queue.get_client(
        partition_id="sync_test",
        num_buckets=2,
        batch_size=10,
    )


def _teardown_sync_client() -> None:
    """Shutdown transfer_queue-managed runtime after sync-client tests."""
    if pul.is_initialized():
        pul.transfer_queue.shutdown()


def test_sync_client_put_and_get():
    """Sync client: incremental put then get complete samples."""
    client = _make_sync_client()
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
        _teardown_sync_client()


def test_sync_client_incomplete_excluded():
    """Sync client: incomplete samples are not returned."""
    client = _make_sync_client()
    try:
        client.put(sample_idx=0, data={"prompt": "hello"})
        # response missing

        samples = client.get(data_fields=["prompt", "response"], batch_size=10)
        assert len(samples) == 0
    finally:
        _teardown_sync_client()


def test_sync_client_batch_size():
    """Sync client: get respects batch_size."""
    client = _make_sync_client()
    try:
        for i in range(5):
            client.put(sample_idx=i, data={"a": i, "b": i})

        samples = client.get(data_fields=["a", "b"], batch_size=3)
        assert len(samples) == 3
    finally:
        _teardown_sync_client()


def test_sync_client_consumption_tracking():
    """Sync client: same task_name does not receive the same sample twice."""
    client = _make_sync_client()
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
        _teardown_sync_client()


def test_sync_client_get_waits_for_batch_completion():
    """Sync client should honor timeout while waiting for more rows."""
    import threading
    import time

    client = _make_sync_client()
    writer_thread = None
    try:
        client.put(sample_idx=0, data={"x": 0})

        def delayed_put():
            time.sleep(0.1)
            client.put(sample_idx=1, data={"x": 1})

        writer_thread = threading.Thread(target=delayed_put, daemon=True)
        writer_thread.start()

        start = time.monotonic()
        samples = client.get(
            data_fields=["x"],
            batch_size=2,
            task_name="sync_wait_batch",
            timeout=0.5,
        )
        elapsed = time.monotonic() - start

        assert elapsed >= 0.09
        assert len(samples) == 2
        assert sorted(row["x"] for row in samples) == [0, 1]
    finally:
        if writer_thread is not None:
            writer_thread.join(timeout=1)
        _teardown_sync_client()


def test_sync_client_get_sample_idxs_preserves_order():
    """Sync client should support sample_idxs reads with stable ordering."""
    client = _make_sync_client()
    try:
        client.put(sample_idx=0, data={"value": "zero"})
        client.put(sample_idx=3, data={"value": "three"})

        rows = client.get(
            data_fields=["value"],
            sample_idxs=[3, 0],
            batch_size=2,
            task_name="sync_sample_idxs",
            timeout=0.2,
        )

        assert [row["sample_idx"] for row in rows] == [3, 0]
        assert [row["value"] for row in rows] == ["three", "zero"]
    finally:
        _teardown_sync_client()


def test_sync_client_get_sample_idxs_duplicate_idx_raises():
    """Sync client should reject duplicate sample_idx values."""
    client = _make_sync_client()
    try:
        with pytest.raises(ValueError, match="duplicate sample_idx"):
            client.get(
                data_fields=["value"],
                sample_idxs=[3, 0, 3],
                task_name="sync_duplicate_lookup",
            )
    finally:
        _teardown_sync_client()


def test_sync_client_get_sample_idxs_batch_size_mismatch_raises():
    """Sync client should reject batch_size mismatches."""
    client = _make_sync_client()
    try:
        with pytest.raises(
            ValueError, match="batch_size must equal len\\(sample_idxs\\)"
        ):
            client.get(
                data_fields=["value"],
                sample_idxs=[3, 0],
                batch_size=1,
                task_name="sync_mismatch_lookup",
            )
    finally:
        _teardown_sync_client()


def test_sync_client_multi_bucket():
    """Sync client: data is sharded across buckets."""
    client = _make_sync_client()
    try:
        # num_buckets=2, so even/odd go to different buckets
        for i in range(6):
            client.put(sample_idx=i, data={"v": i})

        samples = client.get(data_fields=["v"], batch_size=10)
        assert len(samples) == 6
        values = sorted(s["v"] for s in samples)
        assert values == list(range(6))
    finally:
        _teardown_sync_client()


def test_sync_client_clear():
    """Sync client: clear resets all data."""
    client = _make_sync_client()
    try:
        for i in range(3):
            client.put(sample_idx=i, data={"x": i})

        client.clear()

        samples = client.get(data_fields=["x"], batch_size=10)
        assert len(samples) == 0
    finally:
        _teardown_sync_client()


def test_sync_client_put_returns_meta():
    """Sync client: put returns a BatchMeta-compatible dict."""
    client = _make_sync_client()
    try:
        meta = client.put(sample_idx=7, data={"prompt": "hi"})
        assert meta["partition_id"] == "sync_test"
        assert meta["sample_idx"] == 7
        assert "prompt" in meta["fields"]
        assert meta["status"] == "ok"
    finally:
        _teardown_sync_client()


def test_sync_client_partition_id():
    """Sync client: partition_id property is accessible."""
    client = _make_sync_client()
    try:
        assert client.partition_id == "sync_test"
    finally:
        _teardown_sync_client()


@pytest.mark.asyncio
async def test_async_client_auto_initializes_global_system():
    """Async client factory should initialize Pulsing before returning."""
    assert not pul.is_initialized()

    client = await pul.transfer_queue.get_async_client(
        partition_id="auto_async", num_buckets=2, batch_size=10
    )
    assert pul.is_initialized()
    assert await _resolve_local_storage_manager() is not None

    try:
        await client.async_put(sample_idx=0, data={"prompt": "hello"})
        await client.async_put(sample_idx=0, data={"response": "world"})

        samples = await client.async_get(
            data_fields=["prompt", "response"], batch_size=10, task_name="auto"
        )
        assert len(samples) == 1
        assert samples[0] == {"prompt": "hello", "response": "world"}
        assert pul.is_initialized()
    finally:
        if pul.is_initialized():
            await pul.shutdown()


def test_get_client_auto_initializes_and_shutdown_cleans_up():
    """Sync get_client should auto-init and transfer_queue.shutdown should clean up."""
    import pulsing._runtime as runtime

    assert not pul.is_initialized()

    client = pul.transfer_queue.get_client(
        partition_id="auto_sync", num_buckets=2, batch_size=10
    )

    try:
        assert _resolve_local_storage_manager_sync() is not None
        client.put(sample_idx=0, data={"prompt": "hello"})
        client.put(sample_idx=0, data={"response": "world"})

        samples = client.get(
            data_fields=["prompt", "response"], batch_size=10, task_name="auto"
        )
        assert len(samples) == 1
        assert samples[0] == {"prompt": "hello", "response": "world"}
        assert pul.is_initialized()
        assert runtime.owns_system() is True
        assert get_shared_loop() is not None
        assert get_pulsing_loop() is get_shared_loop()
    finally:
        pul.transfer_queue.shutdown()

    assert not pul.is_initialized()
    assert get_shared_loop() is None
    assert runtime.is_cleanup_registered() is True


@pytest.mark.asyncio
async def test_get_client_reuses_explicit_init_loop_without_module_runtime():
    """Sync client should reuse an explicitly initialized global system."""
    import pulsing._runtime as runtime

    await pul.init()

    try:
        client = await asyncio.to_thread(
            pul.transfer_queue.get_client,
            partition_id="reuse_sync",
            num_buckets=2,
            batch_size=10,
        )
        assert await _wait_for_local_storage_manager() is not None

        await asyncio.to_thread(client.put, sample_idx=0, data={"prompt": "hello"})
        await asyncio.to_thread(client.put, sample_idx=0, data={"response": "world"})

        samples = await asyncio.to_thread(
            client.get,
            data_fields=["prompt", "response"],
            batch_size=10,
            task_name="reuse",
        )
        assert len(samples) == 1
        assert samples[0] == {"prompt": "hello", "response": "world"}
        assert get_shared_loop() is None
        assert runtime.owns_system() is False
    finally:
        if pul.is_initialized():
            await pul.shutdown()


@pytest.mark.asyncio
async def test_get_client_same_thread_async_before_init_rejects_direct_call():
    import pulsing._runtime as runtime

    try:
        with pytest.raises(
            RuntimeError, match="cannot be called from an active event loop"
        ):
            pul.transfer_queue.get_client(
                partition_id="before_init_same_thread",
                num_buckets=1,
                batch_size=10,
            )

        assert not pul.is_initialized()
        assert runtime.owns_system() is False
        assert get_shared_loop() is None
        assert get_pulsing_loop() is None
    finally:
        if pul.is_initialized():
            pul.transfer_queue.shutdown()


def test_async_auto_initialized_runtime_can_cleanup_without_public_shutdown():
    """Async auto-init should not require calling pul.transfer_queue.shutdown()."""
    import pulsing._runtime as runtime

    async def main():
        client = await pul.transfer_queue.get_async_client(
            partition_id="auto_async_cleanup", num_buckets=2, batch_size=10
        )
        await client.async_put(sample_idx=0, data={"prompt": "hello"})
        await client.async_put(sample_idx=0, data={"response": "world"})
        rows = await client.async_get(
            data_fields=["prompt", "response"], batch_size=10, task_name="cleanup"
        )
        assert rows == [{"prompt": "hello", "response": "world"}]

    try:
        asyncio.run(main())
        assert pul.is_initialized()
        assert runtime.is_cleanup_registered() is True
        runtime.shutdown()
        assert not pul.is_initialized()
        assert get_shared_loop() is None
    finally:
        if pul.is_initialized():
            runtime.shutdown()
