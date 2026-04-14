"""pulsing.transfer_queue - Fixed-capacity transfer buckets with exact reads.

Each transfer queue topic owns ``num_buckets`` independent buckets. Callers
choose the target bucket explicitly on ``put()``/``get()`` via ``bucket_id``.
Each bucket stores up to ``bucket_capacity`` samples in a ring buffer; once the
bucket is full, inserting a new ``sample_idx`` overwrites the oldest stored
sample in that bucket.

Usage (async)::

    import pulsing as pul

    client = await pul.transfer_queue.get_async_client(
        topic="train", num_buckets=2, bucket_capacity=10
    )
    await client.async_put(
        sample_idx=0,
        data={"prompt": "hello"},
        bucket_id=1,
    )
    await client.async_put(
        sample_idx=0,
        data={"response": "world"},
        bucket_id=1,
    )

    sample = await client.async_get(
        data_fields=["prompt", "response"],
        sample_idx=0,
        bucket_id=1,
        timeout=1.0,
    )

Usage (sync)::

    import pulsing as pul

    client = pul.transfer_queue.get_client(
        topic="train", num_buckets=2, bucket_capacity=10
    )
    client.put(sample_idx=0, data={"prompt": "hello"}, bucket_id=1)
    client.put(sample_idx=0, data={"response": "world"}, bucket_id=1)

    sample = client.get(
        data_fields=["prompt", "response"],
        sample_idx=0,
        bucket_id=1,
        timeout=1.0,
    )

The sync client auto-initializes Pulsing for synchronous callers and
cross-thread use. Async callers running on the active event loop should use
``await get_async_client()`` instead, or move both ``get_client()`` and sync
client calls to another thread. The synchronous client must not be called from
the active Pulsing event loop thread; use ``asyncio.to_thread(...)`` if you
need it from async code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pulsing._async_bridge import get_running_loop as _get_running_loop
from pulsing._async_bridge import run_sync as _run_sync
import pulsing._runtime as _runtime

from .client import AsyncTransferQueueClient, TransferQueueClient


async def _ensure_local_storage_manager() -> None:
    from pulsing.core import get_system
    from .manager import ensure_storage_managers

    await ensure_storage_managers(get_system())


def _ensure_sync_local_storage_manager() -> None:
    _run_sync(_ensure_local_storage_manager())


@dataclass
class BatchMeta:
    """Metadata returned by a put operation."""

    topic: str
    bucket_id: int
    sample_idx: int
    fields: list[str] = field(default_factory=list)
    status: str = "ok"


def shutdown():
    """Optionally shutdown transfer_queue-managed Pulsing runtime, if any."""
    _runtime.shutdown(best_effort=False)


async def get_async_client(
    topic: str,
    num_buckets: int = 1,
    bucket_capacity: int = 10,
) -> AsyncTransferQueueClient:
    """Return an async client for *topic*."""
    await _runtime.ensure_async_runtime()
    await _ensure_local_storage_manager()
    return AsyncTransferQueueClient(
        topic=topic,
        num_buckets=num_buckets,
        bucket_capacity=bucket_capacity,
    )


def get_client(
    topic: str,
    num_buckets: int = 1,
    bucket_capacity: int = 10,
) -> TransferQueueClient:
    """Return a synchronous client for *topic*."""
    if _get_running_loop() is not None:
        raise RuntimeError(
            "get_client() cannot be called from an active event loop. "
            "Use await get_async_client() or call get_client() from "
            "asyncio.to_thread(...)."
        )

    _runtime.ensure_sync_runtime()
    _ensure_sync_local_storage_manager()
    inner = AsyncTransferQueueClient(
        topic=topic,
        num_buckets=num_buckets,
        bucket_capacity=bucket_capacity,
    )
    return TransferQueueClient(inner)


__all__ = [
    "shutdown",
    "get_async_client",
    "get_client",
    "BatchMeta",
    "AsyncTransferQueueClient",
    "TransferQueueClient",
]
