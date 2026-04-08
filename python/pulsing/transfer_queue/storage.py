"""StorageUnit - Per-bucket @remote actor for transfer queue."""

import logging
from typing import Any

from pulsing.core import ActorId, remote

from .backend import TransferBackend

logger = logging.getLogger(__name__)


@remote
class StorageUnit:
    """Per-bucket storage actor wrapping a TransferBackend.

    Args:
        bucket_id: Bucket ID
        batch_size: Default batch size
    """

    def __init__(self, bucket_id: int, batch_size: int = 10):
        self.bucket_id = bucket_id
        self.batch_size = batch_size
        self._backend: TransferBackend | None = None

    def on_start(self, actor_id: ActorId) -> None:
        self._backend = TransferBackend(
            bucket_id=self.bucket_id,
            batch_size=self.batch_size,
        )
        logger.info(f"StorageUnit[{self.bucket_id}] started")

    def on_stop(self) -> None:
        logger.info(f"StorageUnit[{self.bucket_id}] stopping")

    async def put(self, sample_idx: int, data: dict) -> dict:
        """Merge data into sample identified by sample_idx."""
        return await self._backend.put(sample_idx, data)

    async def get_data(
        self,
        fields: list[str],
        batch_size: int,
        task_name: str = "default",
    ) -> list[dict]:
        """Return sample dicts for all fields, optionally filtered by fields."""
        return await self._backend.get_data(fields, batch_size, task_name)

    async def clear(self) -> dict:
        """Reset all state in this bucket."""
        await self._backend.clear()
        return {"status": "ok"}

    async def stats(self) -> dict:
        """Return diagnostic info for this bucket."""
        return await self._backend.stats()
