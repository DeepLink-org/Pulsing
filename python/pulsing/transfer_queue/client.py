"""Transfer queue client - async and sync wrappers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pulsing._async_bridge import get_loop, get_running_loop, run_sync
from pulsing.core import ActorSystem, get_system
from pulsing.core.remote import ActorProxy

from .manager import get_unit_ref

logger = logging.getLogger(__name__)

_GET_POLL_INTERVAL_SECONDS = 0.05


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

    async def _ensure_bucket(self, bucket_id: int) -> ActorProxy:
        self._bind_or_validate_loop()
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

    @staticmethod
    def _validate_sample_idxs(
        sample_idxs: list[int] | None,
        batch_size: int | None,
    ) -> list[int] | None:
        if sample_idxs is None:
            return None

        normalized: list[int] = []
        seen: set[int] = set()
        for sample_idx in sample_idxs:
            if sample_idx in seen:
                raise ValueError(
                    "sample_idxs must not contain duplicate sample_idx values"
                )
            seen.add(sample_idx)
            normalized.append(sample_idx)

        if batch_size is not None and batch_size != len(normalized):
            raise ValueError(
                "batch_size must equal len(sample_idxs) when sample_idxs is provided"
            )

        return normalized

    @staticmethod
    def _deadline_from_timeout(timeout: float | None) -> float | None:
        if timeout is None or timeout <= 0:
            return None
        return time.monotonic() + timeout

    async def _get_batch_once(
        self,
        data_fields: list[str],
        batch_size: int,
        task_name: str,
    ) -> list[dict[str, Any]]:
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

        return collected

    async def _get_requested_samples_once(
        self,
        data_fields: list[str],
        task_name: str,
        pending_by_bucket: dict[int, list[int]],
    ) -> dict[int, dict[str, Any]]:
        resolved: dict[int, dict[str, Any]] = {}

        for bucket_id, sample_idxs in pending_by_bucket.items():
            unit = await self._ensure_bucket(bucket_id)
            rows = await unit.get_data(
                fields=data_fields,
                batch_size=len(sample_idxs),
                task_name=task_name,
                sample_idxs=sample_idxs,
            )
            for row in rows:
                sample_idx = row["sample_idx"]
                resolved[sample_idx] = row

        return resolved

    async def async_put(self, sample_idx: int, data: dict[str, Any]) -> dict[str, Any]:
        """Write (merge) *data* into the sample identified by *sample_idx*.

        Returns a BatchMeta-compatible dict.
        """
        self._bind_or_validate_loop()
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
        sample_idxs: list[int] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Collect complete samples, optionally waiting up to *timeout* seconds.

        When *sample_idxs* is provided, fetch only those samples and preserve the
        caller's requested order. Otherwise, fetch the next unread batch across
        all buckets.

        Raises:
            ValueError: If ``sample_idxs`` contains duplicates, or if both
                ``sample_idxs`` and ``batch_size`` are provided and their lengths
                do not match.
        """
        self._bind_or_validate_loop()
        normalized_sample_idxs = self._validate_sample_idxs(sample_idxs, batch_size)
        deadline = self._deadline_from_timeout(timeout)

        if normalized_sample_idxs is not None:
            pending_by_bucket: dict[int, list[int]] = {}
            for sample_idx in normalized_sample_idxs:
                bucket_id = sample_idx % self.num_buckets
                pending_by_bucket.setdefault(bucket_id, []).append(sample_idx)

            resolved: dict[int, dict[str, Any]] = {}
            while pending_by_bucket:
                resolved.update(
                    await self._get_requested_samples_once(
                        data_fields=data_fields,
                        task_name=task_name,
                        pending_by_bucket=pending_by_bucket,
                    )
                )

                for bucket_id, pending in list(pending_by_bucket.items()):
                    remaining = [
                        sample_idx
                        for sample_idx in pending
                        if sample_idx not in resolved
                    ]
                    if remaining:
                        pending_by_bucket[bucket_id] = remaining
                    else:
                        del pending_by_bucket[bucket_id]

                if not pending_by_bucket or deadline is None:
                    break

                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    break

                await asyncio.sleep(min(_GET_POLL_INTERVAL_SECONDS, remaining_time))

            return [
                resolved[sample_idx]
                for sample_idx in normalized_sample_idxs
                if sample_idx in resolved
            ]

        if batch_size is None:
            batch_size = self.batch_size
        if batch_size <= 0:
            return []

        collected: list[dict[str, Any]] = []
        while len(collected) < batch_size:
            rows = await self._get_batch_once(
                data_fields=data_fields,
                batch_size=batch_size - len(collected),
                task_name=task_name,
            )
            collected.extend(rows)

            if len(collected) >= batch_size or deadline is None:
                break

            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                break

            await asyncio.sleep(min(_GET_POLL_INTERVAL_SECONDS, remaining_time))

        return collected

    async def async_clear(self) -> None:
        """Clear all buckets for this partition."""
        self._bind_or_validate_loop()
        for bucket_id in range(self.num_buckets):
            unit = await self._ensure_bucket(bucket_id)
            await unit.clear()


class TransferQueueClient:
    """Synchronous wrapper around AsyncTransferQueueClient.

    Uses the shared sync bridge so it can be called from synchronous code
    or from another thread while Pulsing runs elsewhere. Async callers on
    the active event loop should use ``get_async_client()`` instead.
    """

    def __init__(self, inner: AsyncTransferQueueClient):
        self._inner = inner

    @property
    def partition_id(self) -> str:
        return self._inner.partition_id

    def put(self, sample_idx: int, data: dict[str, Any]) -> dict[str, Any]:
        return run_sync(self._inner.async_put(sample_idx, data))

    def get(
        self,
        data_fields: list[str],
        batch_size: int | None = None,
        task_name: str = "default",
        sample_idxs: list[int] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        return run_sync(
            self._inner.async_get(
                data_fields=data_fields,
                batch_size=batch_size,
                task_name=task_name,
                sample_idxs=sample_idxs,
                timeout=timeout,
            )
        )

    def clear(self) -> None:
        run_sync(self._inner.async_clear())
