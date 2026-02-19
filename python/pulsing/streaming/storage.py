"""Bucket Storage Actor - Using Pluggable Backend"""

import asyncio
import logging
from typing import Any, AsyncIterator

from pulsing.core import ActorId, StreamMessage, remote

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
            - "memory": Pure in-memory backend (default)
            - Custom name/class: Use register_backend() or pass class
        backend_options: Additional parameters passed to backend
    """

    def __init__(
        self,
        bucket_id: int,
        storage_path: str,
        batch_size: int = 100,
        backend: str | type = "memory",
        backend_options: dict[str, Any] | None = None,
    ):
        self.bucket_id = bucket_id
        self.storage_path = storage_path
        self.batch_size = batch_size
        self._backend_type = backend
        self._backend_options = backend_options or {}

        # Backend instance (initialized in on_start)
        self._backend: StorageBackend | None = None
        self._production_status: dict[int, dict[str, str]] = {}
        self._consumption_status: dict[str, set[int]] = {}
        self._key_to_index: dict[str, int] = {}

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
        before = self._backend.total_count()
        await self._backend.put(record)
        fields = [k for k in record.keys() if not str(k).startswith("_")]
        self._production_status[before] = {field: "ready" for field in fields}
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
        start = self._backend.total_count()
        await self._backend.put_batch(records)
        for i, record in enumerate(records):
            fields = [k for k in record.keys() if not str(k).startswith("_")]
            self._production_status[start + i] = {field: "ready" for field in fields}
        return {"status": "ok", "count": len(records)}

    async def put_tensor(
        self, data: Any, partition_id: str = "default", **kwargs: Any
    ) -> dict:
        if hasattr(self._backend, "put_tensor"):
            meta = await self._backend.put_tensor(
                data, partition_id=partition_id, **kwargs
            )
            if hasattr(meta, "global_indexes") and hasattr(meta, "field_names"):
                for idx in meta.global_indexes:
                    self._production_status[idx] = {
                        field: "ready" for field in meta.field_names
                    }
            return {"status": "ok"}
        raise NotImplementedError("Backend does not support put_tensor")

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

    async def get_meta(
        self,
        fields: list[str],
        batch_size: int,
        task_name: str,
        sampler: Any = None,
        **sampling_kwargs: Any,
    ) -> dict:
        if hasattr(self._backend, "get_meta"):
            meta = await self._backend.get_meta(
                fields=fields,
                batch_size=batch_size,
                task_name=task_name,
                sampler=sampler,
                **sampling_kwargs,
            )
            if hasattr(meta, "to_dict"):
                return meta.to_dict()
            return meta

        consumed = self._consumption_status.setdefault(task_name, set())
        ready = []
        for idx in sorted(self._production_status):
            if idx in consumed:
                continue
            status = self._production_status[idx]
            if all(status.get(field) == "ready" for field in fields):
                ready.append(idx)

        if sampler is not None:
            sampled, marked = sampler.sample(ready, batch_size, **sampling_kwargs)
        else:
            sampled = ready[:batch_size]
            marked = sampled
        consumed.update(marked)
        return {
            "samples": [
                {
                    "partition_id": sampling_kwargs.get("partition_id", "default"),
                    "global_index": idx,
                    "fields": {
                        field: {
                            "name": field,
                            "dtype": None,
                            "shape": None,
                            "production_status": "ready",
                        }
                        for field in fields
                    },
                }
                for idx in sampled
            ],
            "global_indexes": sampled,
        }

    async def get_data(self, batch_meta: dict, fields: list[str] | None = None) -> Any:
        if hasattr(self._backend, "get_data"):
            return await self._backend.get_data(batch_meta, fields=fields)

        indexes = batch_meta.get("global_indexes") or [
            sample.get("global_index", -1) for sample in batch_meta.get("samples", [])
        ]
        if hasattr(self._backend, "get_by_indices"):
            rows = await self._backend.get_by_indices(indexes)
        else:
            rows = []
            for idx in indexes:
                rows.extend(await self._backend.get(limit=1, offset=idx))
        if fields:
            return [{k: v for k, v in row.items() if k in fields} for row in rows]
        return rows

    async def mark_consumed(self, task_name: str, global_indexes: list[int]) -> dict:
        self._consumption_status.setdefault(task_name, set()).update(global_indexes)
        if hasattr(self._backend, "mark_consumed"):
            await self._backend.mark_consumed(task_name, global_indexes)
        return {"status": "ok"}

    async def reset_consumption(self, task_name: str) -> dict:
        self._consumption_status.pop(task_name, None)
        if hasattr(self._backend, "reset_consumption"):
            await self._backend.reset_consumption(task_name)
        return {"status": "ok"}

    async def clear(self, global_indexes: list[int]) -> dict:
        if hasattr(self._backend, "clear"):
            await self._backend.clear(global_indexes)
        return {"status": "ok"}

    async def kv_register(self, key: str, global_index: int) -> dict:
        self._key_to_index[key] = global_index
        return {"status": "ok"}

    async def kv_resolve(self, keys: list[str]) -> dict:
        return {"indexes": [self._key_to_index.get(key, -1) for key in keys]}
