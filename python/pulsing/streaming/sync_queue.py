"""Synchronous queue wrapper - obtained via .sync() method

Call .sync() on async queue to get synchronous wrapper.

Note: Synchronous wrapper is designed for non-async code. Do not use inside async functions,
as it will block the event loop.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from pulsing._async_bridge import bind_run_sync

if TYPE_CHECKING:
    from .queue import Queue, QueueReader


_SAME_LOOP_MESSAGE = (
    "pulsing.streaming sync wrappers cannot block on the same event loop that owns "
    "the underlying queue. Use the async queue API from async code or call the sync "
    "wrapper from another thread."
)
_MISSING_LOOP_MESSAGE = (
    "Event loop not running. Sync wrapper requires a running event loop."
)


class SyncQueue:
    """Synchronous queue wrapper"""

    def __init__(self, queue: "Queue"):
        self._queue = queue
        self._loop = queue._loop
        self._run_sync = bind_run_sync(
            loop=self._loop,
            same_loop="raise",
            same_loop_message=_SAME_LOOP_MESSAGE,
            missing_loop="raise",
            missing_loop_message=_MISSING_LOOP_MESSAGE,
        )

    def put(self, record: dict[str, Any] | list[dict[str, Any]]):
        return self._run_sync(self._queue.put(record))

    def get(
        self,
        bucket_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
        wait: bool = False,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._run_sync(self._queue.get(bucket_id, limit, offset, wait, timeout))

    def flush(self) -> None:
        self._run_sync(self._queue.flush())

    def stats(self) -> dict[str, Any]:
        return self._run_sync(self._queue.stats())


class SyncQueueReader:
    """Synchronous reader wrapper"""

    def __init__(self, reader: "QueueReader"):
        self._reader = reader
        self._loop = reader.queue._loop
        self._run_sync = bind_run_sync(
            loop=self._loop,
            same_loop="raise",
            same_loop_message=_SAME_LOOP_MESSAGE,
            missing_loop="raise",
            missing_loop_message=_MISSING_LOOP_MESSAGE,
        )

    def get(self, limit: int = 100, wait: bool = False, timeout: float | None = None):
        return self._run_sync(self._reader.get(limit, wait, timeout))

    def reset(self) -> None:
        self._reader.reset()

    def set_offset(self, offset: int, bucket_id: int | None = None) -> None:
        self._reader.set_offset(offset, bucket_id)
