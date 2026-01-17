"""Synchronous queue wrapper - obtained via .sync() method

Call .sync() on async queue to get synchronous wrapper.

Note: Synchronous wrapper is designed for non-async code. Do not use inside async functions,
as it will block the event loop.
"""

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .queue import Queue, QueueReader, QueueWriter


class SyncQueue:
    """Synchronous queue wrapper"""

    def __init__(self, queue: "Queue"):
        self._queue = queue
        self._loop = queue._loop

    def _run(self, coro):
        """Run coroutine synchronously"""
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(
                "Event loop not running. Sync wrapper requires a running event loop."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def put(self, record: dict[str, Any] | list[dict[str, Any]]):
        return self._run(self._queue.put(record))

    def get(
        self,
        bucket_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
        wait: bool = False,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._run(self._queue.get(bucket_id, limit, offset, wait, timeout))

    def flush(self) -> None:
        self._run(self._queue.flush())

    def stats(self) -> dict[str, Any]:
        return self._run(self._queue.stats())


class SyncQueueWriter:
    """Synchronous writer wrapper"""

    def __init__(self, writer: "QueueWriter"):
        self._writer = writer
        self._loop = writer.queue._loop

    def _run(self, coro):
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(
                "Event loop not running. Sync wrapper requires a running event loop."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def put(self, record: dict[str, Any] | list[dict[str, Any]]):
        return self._run(self._writer.put(record))

    def flush(self) -> None:
        self._run(self._writer.flush())


class SyncQueueReader:
    """Synchronous reader wrapper"""

    def __init__(self, reader: "QueueReader"):
        self._reader = reader
        self._loop = reader.queue._loop

    def _run(self, coro):
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(
                "Event loop not running. Sync wrapper requires a running event loop."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def get(self, limit: int = 100, wait: bool = False, timeout: float | None = None):
        return self._run(self._reader.get(limit, wait, timeout))

    def reset(self) -> None:
        self._reader.reset()

    def set_offset(self, offset: int, bucket_id: int | None = None) -> None:
        self._reader.set_offset(offset, bucket_id)
