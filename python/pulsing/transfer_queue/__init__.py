"""pulsing.transfer_queue - Data transfer queue for training-inference pipelines.

Supports incremental field-by-field writing to samples keyed by sample_idx,
and batch reading of complete samples.

Usage (async)::

    import pulsing as pul

    client = pul.transfer_queue.get_async_client(
        partition_id="train", num_buckets=2, batch_size=10
    )
    await client.async_put(sample_idx=0, data={"prompt": "hello"})
    await client.async_put(sample_idx=0, data={"response": "world"})

    samples = await client.async_get(
        data_fields=["prompt", "response"], batch_size=2, task_name="train"
    )

Usage (sync)::

    import pulsing as pul

    client = pul.transfer_queue.get_client(partition_id="train", num_buckets=2, batch_size=10)
    client.put(sample_idx=0, data={"prompt": "hello"})
    client.put(sample_idx=0, data={"response": "world"})

    samples = client.get(
        data_fields=["prompt", "response"], batch_size=2, task_name="train"
    )

The sync client auto-initializes Pulsing only for synchronous callers (or
cross-thread use). Same-thread async callers should use ``get_async_client()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulsing._runtime as _runtime

from .client import AsyncTransferQueueClient, TransferQueueClient

_SYNC_AUTO_INIT_SAME_THREAD_MESSAGE = (
    "pulsing.transfer_queue sync auto-init cannot run on the same "
    "thread as a running event loop. Use get_async_client() from "
    "async code or call get_client() from synchronous code or "
    "another thread."
)


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


def get_async_client(
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

    This sync helper is intended for non-async code or for threads other than
    the one currently running the Pulsing event loop.
    """
    _runtime.ensure_sync_runtime(
        same_thread_message=_SYNC_AUTO_INIT_SAME_THREAD_MESSAGE,
    )
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
