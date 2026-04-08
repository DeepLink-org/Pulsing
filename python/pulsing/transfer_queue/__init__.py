"""pulsing.transfer_queue - Data transfer queue for training-inference pipelines.

Supports incremental field-by-field writing to samples keyed by sample_idx,
and batch reading of complete samples.

Usage (async)::

    import pulsing as pul

    await pul.init()

    client = pul.transfer_queue.get_async_client(
        partition_id="train", num_buckets=2, batch_size=10
    )
    await client.async_put(sample_idx=0, data={"prompt": "hello"})
    await client.async_put(sample_idx=0, data={"response": "world"})

    samples = await client.async_get(
        data_fields=["prompt", "response"], batch_size=2, task_name="train"
    )

Usage (sync)::

    import asyncio
    import threading
    import pulsing as pul

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    asyncio.run_coroutine_threadsafe(pul.init(), loop).result()

    client = pul.transfer_queue.get_client(
        partition_id="train", num_buckets=2, batch_size=10, loop=loop
    )
    client.put(sample_idx=0, data={"prompt": "hello"})
    client.put(sample_idx=0, data={"response": "world"})

    samples = client.get(
        data_fields=["prompt", "response"], batch_size=2, task_name="train"
    )

    asyncio.run_coroutine_threadsafe(pul.shutdown(), loop).result()
    loop.call_soon_threadsafe(loop.stop)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field

from .client import AsyncTransferQueueClient, TransferQueueClient

logger = logging.getLogger(__name__)

# Managed background event loop (for sync callers)
_managed_loop: asyncio.AbstractEventLoop | None = None
_managed_thread: threading.Thread | None = None
_managed_lock = threading.Lock()


@dataclass
class BatchMeta:
    """Metadata returned by a put operation."""

    partition_id: str
    sample_idx: int
    fields: list[str] = field(default_factory=list)
    status: str = "ok"


def _start_managed_loop() -> asyncio.AbstractEventLoop:
    """Start a background event loop for sync callers. Idempotent."""
    global _managed_loop, _managed_thread
    with _managed_lock:
        if _managed_thread is not None and _managed_loop is not None:
            return _managed_loop

        ready = threading.Event()

        def _run():
            global _managed_loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _managed_loop = loop
            ready.set()
            loop.run_forever()

        _managed_thread = threading.Thread(
            target=_run, daemon=True, name="pulsing-transfer-queue-loop"
        )
        _managed_thread.start()
        ready.wait()
        return _managed_loop  # type: ignore[return-value]


def shutdown():
    """Shutdown the managed background event loop and thread.

    Only needed when using ``get_client()`` without an explicit *loop*
    argument (i.e. when the module manages its own background loop).
    """
    global _managed_loop, _managed_thread

    if _managed_loop is not None and _managed_loop.is_running():
        _managed_loop.call_soon_threadsafe(_managed_loop.stop)
        if _managed_thread is not None:
            _managed_thread.join(timeout=5)

    with _managed_lock:
        _managed_loop = None
        _managed_thread = None


def get_async_client(
    partition_id: str,
    num_buckets: int = 1,
    batch_size: int = 10,
) -> AsyncTransferQueueClient:
    """Return an async client for *partition_id*.

    Requires ``pul.init()`` to have been called beforehand.

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
    loop: asyncio.AbstractEventLoop | None = None,
) -> TransferQueueClient:
    """Return a synchronous client for *partition_id*.

    Requires ``pul.init()`` to have been called beforehand on the
    event loop that will be used by the client.

    Args:
        partition_id: Logical partition identifier.
        num_buckets: Number of buckets to shard data across.
        batch_size: Default batch size for reads.
        loop: The running event loop to dispatch coroutines on.
              If not provided, starts/reuses a managed background loop.
              Falls back to ``asyncio.get_event_loop()`` if no managed loop exists.
    """
    if loop is None:
        loop = _managed_loop
    if loop is None:
        loop = _start_managed_loop()
    inner = AsyncTransferQueueClient(
        partition_id=partition_id,
        num_buckets=num_buckets,
        batch_size=batch_size,
    )
    return TransferQueueClient(inner, loop)


__all__ = [
    "shutdown",
    "get_async_client",
    "get_client",
    "BatchMeta",
    "AsyncTransferQueueClient",
    "TransferQueueClient",
]
