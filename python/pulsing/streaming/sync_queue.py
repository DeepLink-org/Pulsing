"""Synchronous queue wrappers backed by the global Pulsing sync bridge.

These wrappers must be used from synchronous code or another thread. Do not
call them from the active Pulsing event loop thread; use the async queue API
there, or move sync calls into ``asyncio.to_thread(...)``.
"""

from typing import TYPE_CHECKING, Any

from pulsing._async_bridge import get_loop, run_sync

if TYPE_CHECKING:
    from .queue import Queue, QueueReader


_UNSUPPORTED_SYNC_QUEUE_MESSAGE = (
    "pulsing.streaming sync wrappers require the global Pulsing runtime. "
    "Create the queue from pul.get_system() after await pul.init(), or use the "
    "async queue API. Do not call sync wrappers from the active Pulsing event "
    "loop thread."
)


def _ensure_global_runtime(system) -> None:
    import pulsing as pul

    if not pul.is_initialized():
        raise RuntimeError(_UNSUPPORTED_SYNC_QUEUE_MESSAGE)
    if system is not pul.get_system():
        raise RuntimeError(_UNSUPPORTED_SYNC_QUEUE_MESSAGE)
    if get_loop() is None:
        raise RuntimeError(_UNSUPPORTED_SYNC_QUEUE_MESSAGE)


class SyncQueue:
    """Synchronous queue wrapper."""

    def __init__(self, queue: "Queue"):
        _ensure_global_runtime(queue.system)
        self._system = queue.system
        self._topic = queue.topic
        self._bucket_column = queue.bucket_column
        self._num_buckets = queue.num_buckets
        self._batch_size = queue.batch_size
        self._storage_path = queue.storage_path
        self._backend = queue.backend
        self._backend_options = queue.backend_options

    def _make_queue(self):
        from .queue import Queue

        return Queue(
            system=self._system,
            topic=self._topic,
            bucket_column=self._bucket_column,
            num_buckets=self._num_buckets,
            batch_size=self._batch_size,
            storage_path=self._storage_path,
            backend=self._backend,
            backend_options=self._backend_options,
        )

    def put(self, record: dict[str, Any] | list[dict[str, Any]]):
        async def _do_put():
            return await self._make_queue().put(record)

        return run_sync(_do_put())

    def get(
        self,
        bucket_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
        wait: bool = False,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        async def _do_get():
            return await self._make_queue().get(bucket_id, limit, offset, wait, timeout)

        return run_sync(_do_get())

    def flush(self) -> None:
        async def _do_flush() -> None:
            queue = self._make_queue()
            for bucket_id in range(self._num_buckets):
                bucket = await queue._ensure_bucket(bucket_id)
                await bucket.flush()

        run_sync(_do_flush())

    def stats(self) -> dict[str, Any]:
        async def _do_stats() -> dict[str, Any]:
            queue = self._make_queue()
            bucket_stats = {}
            for bucket_id in range(self._num_buckets):
                bucket = await queue._ensure_bucket(bucket_id)
                bucket_stats[bucket_id] = await bucket.stats()
            return {
                "topic": self._topic,
                "bucket_column": self._bucket_column,
                "num_buckets": self._num_buckets,
                "buckets": bucket_stats,
            }

        return run_sync(_do_stats())


class SyncQueueReader:
    """Synchronous reader wrapper."""

    def __init__(self, reader: "QueueReader"):
        _ensure_global_runtime(reader.queue.system)
        self._system = reader.queue.system
        self._topic = reader.queue.topic
        self._bucket_column = reader.queue.bucket_column
        self._num_buckets = reader.queue.num_buckets
        self._batch_size = reader.queue.batch_size
        self._storage_path = reader.queue.storage_path
        self._backend = reader.queue.backend
        self._backend_options = reader.queue.backend_options
        self._bucket_ids = (
            None if reader.bucket_ids is None else list(reader.bucket_ids)
        )
        self._offsets = dict(reader._offsets)

    def _make_queue(self):
        from .queue import Queue

        return Queue(
            system=self._system,
            topic=self._topic,
            bucket_column=self._bucket_column,
            num_buckets=self._num_buckets,
            batch_size=self._batch_size,
            storage_path=self._storage_path,
            backend=self._backend,
            backend_options=self._backend_options,
        )

    def get(self, limit: int = 100, wait: bool = False, timeout: float | None = None):
        async def _do_get():
            from .queue import QueueReader

            reader = QueueReader(self._make_queue(), bucket_ids=self._bucket_ids)
            reader._offsets = dict(self._offsets)
            records = await reader.get(limit, wait, timeout)
            self._offsets = dict(reader._offsets)
            return records

        return run_sync(_do_get())

    def reset(self) -> None:
        self._offsets.clear()

    def set_offset(self, offset: int, bucket_id: int | None = None) -> None:
        if bucket_id is not None:
            self._offsets[bucket_id] = offset
            return

        bucket_ids = self._bucket_ids
        if bucket_ids is None:
            bucket_ids = list(range(self._num_buckets))
        for bid in bucket_ids:
            self._offsets[bid] = offset
