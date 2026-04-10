"""Transfer queue client - async and sync wrappers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pulsing._runtime import ensure_async_runtime
from pulsing._async_bridge import (
    bind_run_sync,
    get_loop,
    valid_loop,
)
from pulsing.core import ActorSystem, get_system
from pulsing.core.remote import ActorProxy

from .manager import get_unit_ref

logger = logging.getLogger(__name__)


class AsyncTransferQueueClient:
    """Async client for the transfer queue.

    Args:
        partition_id: Logical partition identifier (used as the queue topic)
        num_buckets: Number of buckets to shard across
        batch_size: Default batch size passed to StorageUnit
        system: Explicit ActorSystem. Falls back to the global system if None.
    """

    def __init__(
        self,
        partition_id: str,
        num_buckets: int,
        batch_size: int,
        system: ActorSystem | None = None,
    ):
        self.partition_id = partition_id
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self._system = system
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._auto_init = system is None
        self._runtime_ready = system is not None

        self._bucket_refs: dict[int, ActorProxy] = {}
        self._bucket_locks: dict[int, asyncio.Lock] = {}
        self._bucket_locks_meta = asyncio.Lock()
        self._runtime_lock = asyncio.Lock()

    async def _ensure_runtime(self) -> None:
        if not self._auto_init or self._runtime_ready:
            return

        async with self._runtime_lock:
            if self._runtime_ready:
                return
            await ensure_async_runtime()
            self._runtime_ready = True

    def _get_system(self) -> ActorSystem:
        if self._system is not None:
            return self._system
        return get_system()

    async def _ensure_bucket(self, bucket_id: int) -> ActorProxy:
        await self._ensure_runtime()

        if bucket_id in self._bucket_refs:
            return self._bucket_refs[bucket_id]

        async with self._bucket_locks_meta:
            if bucket_id not in self._bucket_locks:
                self._bucket_locks[bucket_id] = asyncio.Lock()
            lock = self._bucket_locks[bucket_id]

        async with lock:
            if bucket_id in self._bucket_refs:
                return self._bucket_refs[bucket_id]

            system = self._get_system()
            self._bucket_refs[bucket_id] = await get_unit_ref(
                system,
                topic=self.partition_id,
                bucket_id=bucket_id,
                batch_size=self.batch_size,
            )
            logger.debug(
                f"Resolved transfer queue unit {self.partition_id}:{bucket_id}"
            )
            return self._bucket_refs[bucket_id]

    async def async_put(self, sample_idx: int, data: dict[str, Any]) -> dict[str, Any]:
        """Write (merge) *data* into the sample identified by *sample_idx*.

        Returns a BatchMeta-compatible dict.
        """
        bucket_id = sample_idx % self.num_buckets
        unit = await self._ensure_bucket(bucket_id)
        meta = await unit.put(sample_idx=sample_idx, data=data)
        meta["partition_id"] = self.partition_id
        return meta

    async def async_get(
        self,
        data_fields: list[str],
        batch_size: int | None = None,
        task_name: str = "default",
    ) -> list[dict[str, Any]]:
        """Collect up to *batch_size* complete samples across all buckets.

        A sample is "complete" when all *data_fields* have been written.
        """
        batch_size = batch_size or self.batch_size
        collected: list[dict[str, Any]] = []
        remaining = batch_size

        for bucket_id in range(self.num_buckets):
            if remaining <= 0:
                break
            unit = await self._ensure_bucket(bucket_id)
            rows = await unit.get_data(
                fields=data_fields,
                batch_size=remaining,
                task_name=task_name,
            )
            collected.extend(rows)
            remaining -= len(rows)

        return collected[:batch_size]

    async def async_clear(self) -> None:
        """Clear all instantiated buckets."""
        await self._ensure_runtime()
        for bucket_id, unit in list(self._bucket_refs.items()):
            await unit.clear()


class TransferQueueClient:
    """Synchronous wrapper around AsyncTransferQueueClient.

    Uses the shared sync bridge so it can be called from synchronous code
    or from another thread while Pulsing runs elsewhere. Same-thread async
    callers should use ``get_async_client()`` instead.
    """

    def __init__(self, inner: AsyncTransferQueueClient):
        self._inner = inner
        self._run_sync = bind_run_sync(
            loop=lambda: get_loop() or valid_loop(self._inner._loop),
            same_loop="raise",
            same_loop_message=(
                "pulsing.transfer_queue cannot block on the same event loop that "
                "owns the active Pulsing system. Use get_async_client() from async "
                "code or call the sync client from another thread."
            ),
            missing_loop="raise",
            missing_loop_message=(
                "Event loop not running. Sync wrapper requires a running event loop."
            ),
        )

    @property
    def partition_id(self) -> str:
        return self._inner.partition_id

    def put(self, sample_idx: int, data: dict[str, Any]) -> dict[str, Any]:
        return self._run_sync(self._inner.async_put(sample_idx, data))

    def get(
        self,
        data_fields: list[str],
        batch_size: int | None = None,
        task_name: str = "default",
    ) -> list[dict[str, Any]]:
        return self._run_sync(self._inner.async_get(data_fields, batch_size, task_name))

    def clear(self) -> None:
        self._run_sync(self._inner.async_clear())
