"""pulsing.transfer_queue - Data transfer queue for training-inference pipelines.

Supports incremental field-by-field writing to samples keyed by sample_idx,
and batch reading of complete samples.

Usage (async)::

    import pulsing as pul

    client = await pul.transfer_queue.get_async_client(
        partition_id="train", num_buckets=2, batch_size=10
    )
    await client.async_put(sample_idx=0, data={"prompt": "hello"})
    await client.async_put(sample_idx=0, data={"response": "world"})

    samples = await client.async_get(
        data_fields=["prompt", "response"],
        batch_size=2,
        task_name="train",
        timeout=1.0,
    )

    sample = await client.async_get(
        data_fields=["prompt", "response"],
        sample_idxs=[0],
        batch_size=1,
        task_name="trainer_debug",
        timeout=1.0,
    )

Usage (sync)::

    import pulsing as pul

    client = pul.transfer_queue.get_client(partition_id="train", num_buckets=2, batch_size=10)
    client.put(sample_idx=0, data={"prompt": "hello"})
    client.put(sample_idx=0, data={"response": "world"})

    samples = client.get(
        data_fields=["prompt", "response"],
        batch_size=2,
        task_name="train",
        timeout=1.0,
    )

    sample = client.get(
        data_fields=["prompt", "response"],
        sample_idxs=[0],
        batch_size=1,
        task_name="trainer_debug",
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

    partition_id: str
    sample_idx: int
    fields: list[str] = field(default_factory=list)
    status: str = "ok"


def shutdown():
    """Optionally shutdown transfer_queue-managed Pulsing runtime, if any."""
    _runtime.shutdown(best_effort=False)


async def get_async_client(
    partition_id: str,
    num_buckets: int = 1,
    batch_size: int = 10,
) -> AsyncTransferQueueClient:
    """Return an async client for *partition_id*.

    Args:
        partition_id: Logical partition identifier.
        num_buckets: Number of buckets to shard data across.
        batch_size: Default batch size for reads.
    """
    await _runtime.ensure_async_runtime()
    await _ensure_local_storage_manager()
    return AsyncTransferQueueClient(
        partition_id=partition_id,
        num_buckets=num_buckets,
        batch_size=batch_size,
    )


def get_client(
    partition_id: str,
    num_buckets: int = 1,
    batch_size: int = 10,
) -> TransferQueueClient:
    """Return a synchronous client for *partition_id*.

    Args:
        partition_id: Logical partition identifier.
        num_buckets: Number of buckets to shard data across.
        batch_size: Default batch size for reads.

    This helper is intended for synchronous code or for use from another
    thread. Async callers on the active event loop should use
    ``await get_async_client()`` instead, or call ``get_client()`` from
    ``asyncio.to_thread(...)``.
    """
    if _get_running_loop() is not None:
        raise RuntimeError(
            "get_client() cannot be called from an active event loop. "
            "Use await get_async_client() or call get_client() from "
            "asyncio.to_thread(...)."
        )

    _runtime.ensure_sync_runtime()
    _ensure_sync_local_storage_manager()
    inner = AsyncTransferQueueClient(
        partition_id=partition_id,
        num_buckets=num_buckets,
        batch_size=batch_size,
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
