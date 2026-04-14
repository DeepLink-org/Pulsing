"""StorageUnit - per-bucket @remote actor for transfer queue."""

from __future__ import annotations

import logging
from typing import Any

from pulsing.core import ActorId, remote

from .backend import TransferBackend

logger = logging.getLogger(__name__)


@remote
class StorageUnit:
    """Per-bucket storage actor backed by a fixed-capacity ring buffer."""

    def __init__(self, bucket_id: int, bucket_capacity: int = 10):
        self.bucket_id = bucket_id
        self.bucket_capacity = bucket_capacity
        self._backend: TransferBackend | None = None

    def on_start(self, actor_id: ActorId) -> None:
        self._backend = TransferBackend(
            bucket_id=self.bucket_id,
            bucket_capacity=self.bucket_capacity,
        )
        logger.info(f"StorageUnit[{self.bucket_id}] started")

    def on_stop(self) -> None:
        logger.info(f"StorageUnit[{self.bucket_id}] stopping")

    async def put(self, sample_idx: int, data: dict[str, Any]) -> dict[str, Any]:
        """Merge data into the sample identified by *sample_idx*."""
        return await self._backend.put(sample_idx, data)

    async def get_data(
        self,
        fields: list[str],
        sample_idx: int,
    ) -> dict[str, Any] | None:
        """Return one filtered sample if it is present and complete."""
        return await self._backend.get_data(fields, sample_idx)

    async def clear(self) -> dict[str, str]:
        """Reset all state in this bucket."""
        await self._backend.clear()
        return {"status": "ok"}

    async def get_config(self) -> dict[str, int]:
        """Expose configuration so managers can detect mismatched reuse."""
        return {
            "bucket_id": self.bucket_id,
            "bucket_capacity": self.bucket_capacity,
        }

    async def stats(self) -> dict[str, Any]:
        """Return diagnostic info for this bucket."""
        return await self._backend.stats()
