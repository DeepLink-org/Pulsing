"""Bucket Storage Actor - Using Pluggable Backend"""

import asyncio
import logging
from typing import Any, AsyncIterator

from pulsing.actor import ActorId, StreamMessage, remote

from .backend import StorageBackend, get_backend_class

logger = logging.getLogger(__name__)


@remote
class BucketStorage:
    """Storage Actor for a Single Bucket

    Uses pluggable StorageBackend for data storage.

    Args:
        bucket_id: Bucket ID
        storage_path: Storage path
        batch_size: Batch size
        backend: Backend name or backend class
            - "memory": Pure in-memory backend
            - "lance": Lance persistent backend (default)
            - Custom class: Class implementing StorageBackend protocol
        backend_options: Additional parameters passed to backend
    """

    def __init__(
        self,
        bucket_id: int,
        storage_path: str,
        batch_size: int = 100,
        backend: str | type = "lance",
        backend_options: dict[str, Any] | None = None,
    ):
        self.bucket_id = bucket_id
        self.storage_path = storage_path
        self.batch_size = batch_size
        self._backend_type = backend
        self._backend_options = backend_options or {}

        # Backend instance (initialized in on_start)
        self._backend: StorageBackend | None = None

    def on_start(self, actor_id: ActorId) -> None:
        # Create backend instance
        backend_class = get_backend_class(self._backend_type)
        self._backend = backend_class(
            bucket_id=self.bucket_id,
            storage_path=self.storage_path,
            batch_size=self.batch_size,
            **self._backend_options,
        )
        backend_name = getattr(backend_class, "__name__", str(self._backend_type))
        logger.info(
            f"BucketStorage[{self.bucket_id}] started with {backend_name} at {self.storage_path}"
        )

    def on_stop(self) -> None:
        logger.info(f"BucketStorage[{self.bucket_id}] stopping")

    # ========== Public Remote Methods ==========

    async def put(self, record: dict) -> dict:
        """Put a single record.

        Args:
            record: Record to store

        Returns:
            {"status": "ok"}
        """
        if not record:
            raise ValueError("Missing 'record'")
        await self._backend.put(record)
        return {"status": "ok"}

    async def put_batch(self, records: list[dict]) -> dict:
        """Put multiple records.

        Args:
            records: List of records to store

        Returns:
            {"status": "ok", "count": N}
        """
        if not records:
            raise ValueError("Missing 'records'")
        await self._backend.put_batch(records)
        return {"status": "ok", "count": len(records)}

    async def get(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Get records.

        Args:
            limit: Maximum number of records to return
            offset: Starting offset

        Returns:
            List of records
        """
        return await self._backend.get(limit, offset)

    async def get_stream(
        self,
        limit: int = 100,
        offset: int = 0,
        wait: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[list[dict]]:
        """Get records as a stream.

        Args:
            limit: Maximum number of records to return
            offset: Starting offset
            wait: Whether to wait for new records
            timeout: Timeout in seconds

        Yields:
            Batches of records
        """
        async for records in self._backend.get_stream(limit, offset, wait, timeout):
            yield records

    async def flush(self) -> dict:
        """Flush pending writes.

        Returns:
            {"status": "ok"}
        """
        await self._backend.flush()
        return {"status": "ok"}

    async def stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Statistics dict from backend
        """
        return await self._backend.stats()
