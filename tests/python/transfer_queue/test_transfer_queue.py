"""Tests for the Pulsing transfer_queue exact-read API."""

from __future__ import annotations

import asyncio

import pytest

import pulsing as pul
from pulsing._async_bridge import get_pulsing_loop, get_shared_loop, run_sync
from pulsing.exceptions import PulsingActorError
from pulsing.transfer_queue.backend import TransferBackend
from pulsing.transfer_queue.client import AsyncTransferQueueClient
from pulsing.transfer_queue.manager import STORAGE_MANAGER_NAME, StorageManager
from pulsing.transfer_queue.storage import StorageUnit


@pytest.fixture
async def actor_system():
    """Create a standalone actor system for testing."""
    system = await pul.actor_system()
    yield system
    await system.shutdown()


@pytest.fixture
def backend():
    """Create a TransferBackend for direct testing."""
    return TransferBackend(bucket_id=0, bucket_capacity=2)


def test_backend_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="bucket_capacity must be greater than 0"):
        TransferBackend(bucket_id=0, bucket_capacity=0)


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
    queue_client = AsyncTransferQueueClient(
        topic="test",
        num_buckets=2,
        bucket_capacity=2,
        system=actor_system._inner,
    )
    yield queue_client
    await queue_client.async_clear()


# ============================================================================
# TransferBackend Unit Tests
# ============================================================================


@pytest.mark.asyncio
async def test_backend_put_creates_sample(backend):
    meta = await backend.put(0, {"prompt": "hello"})
    assert meta["bucket_id"] == 0
    assert meta["sample_idx"] == 0
    assert meta["fields"] == ["prompt"]
    assert meta["status"] == "ok"


@pytest.mark.asyncio
async def test_backend_put_merges_existing_sample(backend):
    await backend.put(0, {"prompt": "hello"})
    meta = await backend.put(0, {"response": "world"})
    assert sorted(meta["fields"]) == ["prompt", "response"]


@pytest.mark.asyncio
async def test_backend_get_data_requires_all_fields(backend):
    await backend.put(0, {"prompt": "hello"})
    assert await backend.get_data(fields=["prompt", "response"], sample_idx=0) is None

    await backend.put(0, {"response": "world"})
    assert await backend.get_data(
        fields=["prompt", "response"],
        sample_idx=0,
    ) == {"prompt": "hello", "response": "world"}


@pytest.mark.asyncio
async def test_backend_get_data_returns_filtered_fields(backend):
    await backend.put(0, {"prompt": "hello", "response": "world", "extra": 42})
    assert await backend.get_data(fields=["response"], sample_idx=0) == {
        "response": "world"
    }


@pytest.mark.asyncio
async def test_backend_overwrites_oldest_when_full(backend):
    await backend.put(0, {"value": "oldest"})
    await backend.put(1, {"value": "middle"})
    await backend.put(2, {"value": "newest"})

    assert await backend.get_data(fields=["value"], sample_idx=0) is None
    assert await backend.get_data(fields=["value"], sample_idx=1) == {"value": "middle"}
    assert await backend.get_data(fields=["value"], sample_idx=2) == {"value": "newest"}


@pytest.mark.asyncio
async def test_backend_updating_existing_sample_does_not_refresh_fifo(backend):
    await backend.put(0, {"value": "oldest"})
    await backend.put(1, {"value": "middle"})
    await backend.put(0, {"extra": "still-oldest"})
    await backend.put(2, {"value": "newest"})

    assert await backend.get_data(fields=["value"], sample_idx=0) is None
    assert await backend.get_data(fields=["value"], sample_idx=1) == {"value": "middle"}


@pytest.mark.asyncio
async def test_backend_clear(backend):
    await backend.put(0, {"a": 1})
    await backend.clear()
    assert await backend.get_data(fields=["a"], sample_idx=0) is None


@pytest.mark.asyncio
async def test_backend_stats(backend):
    await backend.put(0, {"a": 1})
    await backend.put(1, {"a": 2})
    stats = await backend.stats()
    assert stats["bucket_id"] == 0
    assert stats["bucket_capacity"] == 2
    assert stats["sample_count"] == 2
    assert stats["backend"] == "transfer_ring_buffer"


@pytest.mark.asyncio
async def test_backend_rejects_empty_field_requests(backend):
    with pytest.raises(ValueError, match="data_fields must not be empty"):
        await backend.get_data(fields=[], sample_idx=0)


@pytest.mark.asyncio
async def test_backend_recovers_from_missing_slot_entry(backend):
    backend._sample_to_slot[3] = 0
    backend._slots[0] = None

    meta = await backend.put(3, {"value": "recovered"})

    assert meta["sample_idx"] == 3
    assert await backend.get_data(fields=["value"], sample_idx=3) == {
        "value": "recovered"
    }


@pytest.mark.asyncio
async def test_backend_get_data_rejects_stale_slot_mapping(backend):
    backend._sample_to_slot[9] = 0
    backend._slots[0] = backend._store_new_sample(1, {"value": "other"})

    assert await backend.get_data(fields=["value"], sample_idx=9) is None


# ============================================================================
# StorageUnit Actor Tests
# ============================================================================


@pytest.fixture
async def storage_unit(actor_system):
    proxy = await StorageUnit.spawn(
        bucket_id=0,
        bucket_capacity=2,
        system=actor_system._inner,
        name="test_storage_unit_0",
        public=True,
    )
    yield proxy


@pytest.mark.asyncio
async def test_storage_unit_put_and_get(storage_unit):
    await storage_unit.put(sample_idx=0, data={"prompt": "hello"})
    await storage_unit.put(sample_idx=0, data={"response": "world"})
    row = await storage_unit.get_data(
        fields=["prompt", "response"],
        sample_idx=0,
    )
    assert row == {"prompt": "hello", "response": "world"}


@pytest.mark.asyncio
async def test_storage_unit_get_config(storage_unit):
    config = await storage_unit.get_config()
    assert config == {"bucket_id": 0, "bucket_capacity": 2}


@pytest.mark.asyncio
async def test_storage_unit_clear(storage_unit):
    await storage_unit.put(sample_idx=0, data={"x": 1})
    result = await storage_unit.clear()
    assert result["status"] == "ok"
    assert await storage_unit.get_data(fields=["x"], sample_idx=0) is None


# ============================================================================
# AsyncTransferQueueClient Tests
# ============================================================================


@pytest.mark.asyncio
async def test_client_put_and_get_exact_bucket(client):
    await client.async_put(sample_idx=0, data={"prompt": "hello"}, bucket_id=1)
    await client.async_put(sample_idx=0, data={"response": "world"}, bucket_id=1)

    row = await client.async_get(
        data_fields=["prompt", "response"],
        sample_idx=0,
        bucket_id=1,
    )
    assert row == {"prompt": "hello", "response": "world"}


@pytest.mark.asyncio
async def test_client_get_wrong_bucket_returns_none(client):
    await client.async_put(sample_idx=0, data={"value": "bucket-1"}, bucket_id=1)
    row = await client.async_get(
        data_fields=["value"],
        sample_idx=0,
        bucket_id=0,
        timeout=0.1,
    )
    assert row is None


@pytest.mark.asyncio
async def test_client_get_waits_for_missing_fields(client):
    import time

    await client.async_put(sample_idx=0, data={"prompt": "hello"}, bucket_id=0)

    async def delayed_put():
        await asyncio.sleep(0.1)
        await client.async_put(sample_idx=0, data={"response": "world"}, bucket_id=0)

    producer = asyncio.create_task(delayed_put())
    start = time.monotonic()
    row = await client.async_get(
        data_fields=["prompt", "response"],
        sample_idx=0,
        bucket_id=0,
        timeout=0.5,
    )
    elapsed = time.monotonic() - start
    await producer

    assert elapsed >= 0.09
    assert row == {"prompt": "hello", "response": "world"}


@pytest.mark.asyncio
async def test_client_get_timeout_returns_none(client):
    import time

    await client.async_put(sample_idx=0, data={"prompt": "hello"}, bucket_id=0)
    start = time.monotonic()
    row = await client.async_get(
        data_fields=["prompt", "response"],
        sample_idx=0,
        bucket_id=0,
        timeout=0.3,
    )
    elapsed = time.monotonic() - start

    assert elapsed >= 0.25
    assert row is None


@pytest.mark.asyncio
async def test_client_bucket_validation(client):
    with pytest.raises(ValueError, match="bucket_id must be in"):
        await client.async_put(sample_idx=0, data={"x": 1}, bucket_id=2)

    with pytest.raises(ValueError, match="bucket_id must be in"):
        await client.async_get(data_fields=["x"], sample_idx=0, bucket_id=-1)


@pytest.mark.asyncio
async def test_client_clear(client):
    await client.async_put(sample_idx=0, data={"x": 0}, bucket_id=0)
    await client.async_put(sample_idx=1, data={"x": 1}, bucket_id=1)
    await client.async_clear()
    assert await client.async_get(data_fields=["x"], sample_idx=0, bucket_id=0) is None
    assert await client.async_get(data_fields=["x"], sample_idx=1, bucket_id=1) is None


@pytest.mark.asyncio
async def test_client_put_returns_meta(client):
    meta = await client.async_put(sample_idx=42, data={"prompt": "hi"}, bucket_id=1)
    assert meta["topic"] == "test"
    assert meta["bucket_id"] == 1
    assert meta["sample_idx"] == 42
    assert "prompt" in meta["fields"]
    assert meta["status"] == "ok"


@pytest.mark.asyncio
async def test_client_multi_bucket_isolation(client):
    await client.async_put(sample_idx=7, data={"value": "left"}, bucket_id=0)
    await client.async_put(sample_idx=7, data={"value": "right"}, bucket_id=1)

    left = await client.async_get(data_fields=["value"], sample_idx=7, bucket_id=0)
    right = await client.async_get(data_fields=["value"], sample_idx=7, bucket_id=1)
    assert left == {"value": "left"}
    assert right == {"value": "right"}


@pytest.mark.asyncio
async def test_client_ring_buffer_overwrite(client):
    await client.async_put(sample_idx=0, data={"value": "oldest"}, bucket_id=0)
    await client.async_put(sample_idx=1, data={"value": "middle"}, bucket_id=0)
    await client.async_put(sample_idx=2, data={"value": "newest"}, bucket_id=0)

    assert (
        await client.async_get(
            data_fields=["value"],
            sample_idx=0,
            bucket_id=0,
            timeout=0.1,
        )
        is None
    )
    assert await client.async_get(
        data_fields=["value"],
        sample_idx=2,
        bucket_id=0,
    ) == {"value": "newest"}


@pytest.mark.asyncio
async def test_client_concurrent_writes(client):
    async def write(idx: int):
        await client.async_put(
            sample_idx=idx,
            data={"value": idx},
            bucket_id=idx % 2,
        )

    await asyncio.gather(*(write(i) for i in range(4)))
    for idx in range(4):
        row = await client.async_get(
            data_fields=["value"],
            sample_idx=idx,
            bucket_id=idx % 2,
        )
        assert row == {"value": idx}


@pytest.mark.asyncio
async def test_client_capacity_mismatch_raises(actor_system):
    from pulsing.transfer_queue.manager import ensure_storage_managers

    await ensure_storage_managers(actor_system._inner)
    client_a = AsyncTransferQueueClient(
        topic="cfg",
        num_buckets=1,
        bucket_capacity=1,
        system=actor_system._inner,
    )
    client_b = AsyncTransferQueueClient(
        topic="cfg",
        num_buckets=1,
        bucket_capacity=3,
        system=actor_system._inner,
    )

    await client_a.async_put(sample_idx=0, data={"x": 1}, bucket_id=0)
    with pytest.raises(ValueError, match="different bucket_capacity"):
        await client_b.async_put(sample_idx=1, data={"x": 2}, bucket_id=0)

    await client_a.async_clear()


def test_client_init_validation():
    with pytest.raises(ValueError, match="num_buckets must be greater than 0"):
        AsyncTransferQueueClient(topic="bad", num_buckets=0, bucket_capacity=1)

    with pytest.raises(ValueError, match="bucket_capacity must be greater than 0"):
        AsyncTransferQueueClient(topic="bad", num_buckets=1, bucket_capacity=0)


def test_client_methods_require_event_loop():
    client = AsyncTransferQueueClient(
        topic="sync_only", num_buckets=1, bucket_capacity=1
    )

    with pytest.raises(RuntimeError, match="must run inside an event loop"):
        client._bind_or_validate_loop()


@pytest.mark.asyncio
async def test_client_binds_to_first_running_loop_when_created_sync(actor_system):
    from pulsing.transfer_queue.manager import ensure_storage_managers

    await ensure_storage_managers(actor_system._inner)
    client = await asyncio.to_thread(
        AsyncTransferQueueClient,
        "late_bound",
        1,
        2,
        actor_system._inner,
    )

    assert client._bound_loop is None

    await client.async_put(sample_idx=0, data={"value": "ok"}, bucket_id=0)

    assert client._bound_loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_client_rejects_different_event_loop(client):
    with pytest.raises(RuntimeError, match="bound to a different event loop"):
        await asyncio.to_thread(asyncio.run, client.async_clear())


@pytest.mark.asyncio
async def test_client_reraises_unrelated_actor_errors(actor_system, monkeypatch):
    async def fail_get_unit_ref(*args, **kwargs):
        raise PulsingActorError("unexpected failure")

    monkeypatch.setattr("pulsing.transfer_queue.client.get_unit_ref", fail_get_unit_ref)
    queue_client = AsyncTransferQueueClient(
        topic="test",
        num_buckets=1,
        bucket_capacity=2,
        system=actor_system._inner,
    )

    with pytest.raises(PulsingActorError, match="unexpected failure"):
        await queue_client._ensure_bucket(0)


# ============================================================================
# TransferQueueClient (Sync Wrapper) Tests
# ============================================================================


def _make_sync_client(topic: str = "sync_test", bucket_capacity: int = 2):
    if pul.is_initialized():
        pul.transfer_queue.shutdown()
    assert not pul.is_initialized()
    return pul.transfer_queue.get_client(
        topic=topic,
        num_buckets=2,
        bucket_capacity=bucket_capacity,
    )


def _teardown_sync_client() -> None:
    if pul.is_initialized():
        pul.transfer_queue.shutdown()


def test_sync_client_put_and_get():
    client = _make_sync_client()
    try:
        client.put(sample_idx=0, data={"prompt": "hello"}, bucket_id=1)
        client.put(sample_idx=0, data={"response": "world"}, bucket_id=1)
        row = client.get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=1,
        )
        assert row == {"prompt": "hello", "response": "world"}
    finally:
        _teardown_sync_client()


def test_sync_client_waits_for_fields():
    import threading
    import time

    client = _make_sync_client()
    writer_thread = None
    try:
        client.put(sample_idx=0, data={"prompt": "hello"}, bucket_id=0)

        def delayed_put():
            time.sleep(0.1)
            client.put(sample_idx=0, data={"response": "world"}, bucket_id=0)

        writer_thread = threading.Thread(target=delayed_put, daemon=True)
        writer_thread.start()

        start = time.monotonic()
        row = client.get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=0,
            timeout=0.5,
        )
        elapsed = time.monotonic() - start

        assert elapsed >= 0.09
        assert row == {"prompt": "hello", "response": "world"}
    finally:
        if writer_thread is not None:
            writer_thread.join(timeout=1)
        _teardown_sync_client()


def test_sync_client_timeout_returns_none():
    client = _make_sync_client()
    try:
        client.put(sample_idx=0, data={"prompt": "hello"}, bucket_id=0)
        row = client.get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=0,
            timeout=0.1,
        )
        assert row is None
    finally:
        _teardown_sync_client()


def test_sync_client_wrong_bucket_returns_none():
    client = _make_sync_client()
    try:
        client.put(sample_idx=0, data={"value": "bucket-1"}, bucket_id=1)
        row = client.get(
            data_fields=["value"],
            sample_idx=0,
            bucket_id=0,
            timeout=0.1,
        )
        assert row is None
    finally:
        _teardown_sync_client()


def test_sync_client_bucket_validation():
    client = _make_sync_client()
    try:
        with pytest.raises(ValueError, match="bucket_id must be in"):
            client.put(sample_idx=0, data={"x": 1}, bucket_id=2)

        with pytest.raises(ValueError, match="bucket_id must be in"):
            client.get(data_fields=["x"], sample_idx=0, bucket_id=-1)
    finally:
        _teardown_sync_client()


def test_sync_client_clear():
    client = _make_sync_client()
    try:
        client.put(sample_idx=0, data={"x": 0}, bucket_id=0)
        client.put(sample_idx=1, data={"x": 1}, bucket_id=1)
        client.clear()
        assert client.get(data_fields=["x"], sample_idx=0, bucket_id=0) is None
        assert client.get(data_fields=["x"], sample_idx=1, bucket_id=1) is None
    finally:
        _teardown_sync_client()


def test_sync_client_put_returns_meta():
    client = _make_sync_client()
    try:
        meta = client.put(sample_idx=7, data={"prompt": "hi"}, bucket_id=1)
        assert meta["topic"] == "sync_test"
        assert meta["bucket_id"] == 1
        assert meta["sample_idx"] == 7
        assert "prompt" in meta["fields"]
    finally:
        _teardown_sync_client()


def test_sync_client_topic_property():
    client = _make_sync_client()
    try:
        assert client.topic == "sync_test"
    finally:
        _teardown_sync_client()


# ============================================================================
# Runtime / Factory Tests
# ============================================================================


@pytest.mark.asyncio
async def test_async_client_auto_initializes_global_system():
    assert not pul.is_initialized()

    client = await pul.transfer_queue.get_async_client(
        topic="auto_async",
        num_buckets=2,
        bucket_capacity=2,
    )
    assert pul.is_initialized()
    assert await _resolve_local_storage_manager() is not None

    try:
        await client.async_put(sample_idx=0, data={"prompt": "hello"}, bucket_id=0)
        await client.async_put(sample_idx=0, data={"response": "world"}, bucket_id=0)
        row = await client.async_get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=0,
        )
        assert row == {"prompt": "hello", "response": "world"}
        assert pul.is_initialized()
    finally:
        if pul.is_initialized():
            await pul.shutdown()


def test_get_client_auto_initializes_and_shutdown_cleans_up():
    import pulsing._runtime as runtime

    assert not pul.is_initialized()

    client = pul.transfer_queue.get_client(
        topic="auto_sync",
        num_buckets=2,
        bucket_capacity=2,
    )

    try:
        assert _resolve_local_storage_manager_sync() is not None
        client.put(sample_idx=0, data={"prompt": "hello"}, bucket_id=0)
        client.put(sample_idx=0, data={"response": "world"}, bucket_id=0)
        row = client.get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=0,
        )
        assert row == {"prompt": "hello", "response": "world"}
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
    import pulsing._runtime as runtime

    await pul.init()

    try:
        client = await asyncio.to_thread(
            pul.transfer_queue.get_client,
            topic="reuse_sync",
            num_buckets=2,
            bucket_capacity=2,
        )
        assert await _wait_for_local_storage_manager() is not None

        await asyncio.to_thread(
            client.put, sample_idx=0, data={"prompt": "hello"}, bucket_id=0
        )
        await asyncio.to_thread(
            client.put, sample_idx=0, data={"response": "world"}, bucket_id=0
        )

        row = await asyncio.to_thread(
            client.get,
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=0,
        )
        assert row == {"prompt": "hello", "response": "world"}
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
                topic="before_init_same_thread",
                num_buckets=1,
                bucket_capacity=2,
            )

        assert not pul.is_initialized()
        assert runtime.owns_system() is False
        assert get_shared_loop() is None
        assert get_pulsing_loop() is None
    finally:
        if pul.is_initialized():
            pul.transfer_queue.shutdown()


def test_async_auto_initialized_runtime_can_cleanup_without_public_shutdown():
    import pulsing._runtime as runtime

    async def main():
        client = await pul.transfer_queue.get_async_client(
            topic="auto_async_cleanup",
            num_buckets=2,
            bucket_capacity=2,
        )
        await client.async_put(sample_idx=0, data={"prompt": "hello"}, bucket_id=0)
        await client.async_put(sample_idx=0, data={"response": "world"}, bucket_id=0)
        row = await client.async_get(
            data_fields=["prompt", "response"],
            sample_idx=0,
            bucket_id=0,
        )
        assert row == {"prompt": "hello", "response": "world"}

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
