"""Transfer queue client - async and sync exact-read wrappers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pulsing._async_bridge import get_loop, get_running_loop, run_sync
from pulsing.core import ActorSystem, get_system
from pulsing.core.remote import ActorProxy
from pulsing.exceptions import PulsingActorError

from .manager import get_unit_ref

logger = logging.getLogger(__name__)

_GET_POLL_INTERVAL_SECONDS = 0.05


class AsyncTransferQueueClient:
    """Async client for the transfer queue exact-read API."""

    def __init__(
        self,
        topic: str,
        num_buckets: int,
        bucket_capacity: int,
        system: ActorSystem | None = None,
    ):
        if num_buckets <= 0:
            raise ValueError("num_buckets must be greater than 0")
        if bucket_capacity <= 0:
            raise ValueError("bucket_capacity must be greater than 0")

        self.topic = topic
        self.num_buckets = num_buckets
        self.bucket_capacity = bucket_capacity
        self._system = system

        self._bound_loop = get_running_loop() or get_loop()
        self._bucket_refs: dict[int, ActorProxy] = {}
        self._bucket_locks: dict[int, asyncio.Lock] = {}
        self._bucket_locks_meta = asyncio.Lock()

    def _get_system(self) -> ActorSystem:
        if self._system is not None:
            return self._system
        return get_system()

    def _bind_or_validate_loop(self) -> None:
        running_loop = get_running_loop()
        if running_loop is None:
            raise RuntimeError(
                "AsyncTransferQueueClient methods must run inside an event loop."
            )

        if self._bound_loop is None:
            self._bound_loop = running_loop
            return

        if running_loop is not self._bound_loop:
            raise RuntimeError(
                "AsyncTransferQueueClient is bound to a different event loop. "
                "Create a new client in the current loop, or use get_async_client() "
                "instead of TransferQueueClient from async code."
            )

    def _validate_bucket_id(self, bucket_id: int) -> None:
        if not 0 <= bucket_id < self.num_buckets:
            raise ValueError(
                f"bucket_id must be in [0, {self.num_buckets - 1}], got {bucket_id}"
            )

    async def _ensure_bucket(self, bucket_id: int) -> ActorProxy:
        self._bind_or_validate_loop()
        self._validate_bucket_id(bucket_id)

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
            try:
                self._bucket_refs[bucket_id] = await get_unit_ref(
                    system,
                    topic=self.topic,
                    bucket_id=bucket_id,
                    bucket_capacity=self.bucket_capacity,
                )
            except PulsingActorError as exc:
                if "different bucket_capacity" in str(exc):
                    raise ValueError(str(exc)) from exc
                raise
            logger.debug(f"Resolved transfer queue unit {self.topic}:{bucket_id}")
            return self._bucket_refs[bucket_id]

    @staticmethod
    def _deadline_from_timeout(timeout: float | None) -> float | None:
        if timeout is None or timeout <= 0:
            return None
        return time.monotonic() + timeout

    async def async_put(
        self,
        sample_idx: int,
        data: dict[str, Any],
        bucket_id: int = 0,
    ) -> dict[str, Any]:
        """Write (merge) *data* into one sample in the selected bucket."""
        self._bind_or_validate_loop()
        unit = await self._ensure_bucket(bucket_id)
        meta = await unit.put(sample_idx=sample_idx, data=data)
        meta["topic"] = self.topic
        meta["bucket_id"] = bucket_id
        return meta

    async def async_get(
        self,
        data_fields: list[str],
        sample_idx: int,
        bucket_id: int = 0,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Read one sample from one bucket, optionally waiting up to *timeout*."""
        self._bind_or_validate_loop()
        unit = await self._ensure_bucket(bucket_id)
        deadline = self._deadline_from_timeout(timeout)

        while True:
            row = await unit.get_data(fields=data_fields, sample_idx=sample_idx)
            if row is not None:
                return row

            if deadline is None:
                return None

            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                return None

            await asyncio.sleep(min(_GET_POLL_INTERVAL_SECONDS, remaining_time))

    async def async_clear(self) -> None:
        """Clear all buckets for this topic."""
        self._bind_or_validate_loop()
        for bucket_id in range(self.num_buckets):
            unit = await self._ensure_bucket(bucket_id)
            await unit.clear()


class TransferQueueClient:
    """Synchronous wrapper around AsyncTransferQueueClient."""

    def __init__(self, inner: AsyncTransferQueueClient):
        self._inner = inner

    @property
    def topic(self) -> str:
        return self._inner.topic

    def put(
        self,
        sample_idx: int,
        data: dict[str, Any],
        bucket_id: int = 0,
    ) -> dict[str, Any]:
        return run_sync(self._inner.async_put(sample_idx, data, bucket_id=bucket_id))

    def get(
        self,
        data_fields: list[str],
        sample_idx: int,
        bucket_id: int = 0,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return run_sync(
            self._inner.async_get(
                data_fields=data_fields,
                sample_idx=sample_idx,
                bucket_id=bucket_id,
                timeout=timeout,
            )
        )

    def clear(self) -> None:
        run_sync(self._inner.async_clear())
