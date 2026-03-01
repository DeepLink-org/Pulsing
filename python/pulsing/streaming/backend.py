"""Storage Backend Protocol - Pluggable Storage Implementation

Defines StorageBackend protocol for pluggable storage:
- MemoryBackend: Pure in-memory (built-in default, no extra deps)
- Custom backends: register via register_backend() or pass class to write_queue()

Usage:
    # Built-in memory backend
    writer = await write_queue(system, "topic", backend="memory")

    # Custom backend (e.g. from a plugin package)
    from some_plugin import MyBackend
    from pulsing.streaming import register_backend
    register_backend("my_backend", MyBackend)
    writer = await write_queue(system, "topic", backend="my_backend")

    # Or pass class directly
    writer = await write_queue(system, "topic", backend=MyBackend)
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Any, AsyncIterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def build_batch_meta(
    sampled: list[int], fields: list[str], partition_id: str = "default"
) -> dict:
    """Build the standard batch-meta dict used by get_meta implementations."""
    return {
        "samples": [
            {
                "partition_id": partition_id,
                "global_index": idx,
                "fields": {
                    f: {
                        "name": f,
                        "dtype": None,
                        "shape": None,
                        "production_status": "ready",
                    }
                    for f in fields
                },
            }
            for idx in sampled
        ],
        "global_indexes": sampled,
    }


@runtime_checkable
class StorageBackend(Protocol):
    """Core Storage Backend Protocol.

    Every backend must implement these seven methods.
    Duck typing is fine — inheritance from this class is not required.
    """

    @abstractmethod
    async def put(self, record: dict[str, Any]) -> None:
        """Write a single record."""
        ...

    @abstractmethod
    async def put_batch(self, records: list[dict[str, Any]]) -> None:
        """Write records in batch."""
        ...

    @abstractmethod
    async def get(self, limit: int, offset: int) -> list[dict[str, Any]]:
        """Read records."""
        ...

    @abstractmethod
    async def get_stream(
        self,
        limit: int,
        offset: int,
        wait: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Stream records."""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Flush buffer to persistent storage."""
        ...

    @abstractmethod
    async def stats(self) -> dict[str, Any]:
        """Return statistics dict."""
        ...

    @abstractmethod
    def total_count(self) -> int:
        """Total record count (synchronous)."""
        ...


@runtime_checkable
class TensorBackend(Protocol):
    """Extension protocol for tensor-native backends.

    Backends that store tensor/array data efficiently should implement this.
    BucketStorage checks ``isinstance(backend, TensorBackend)`` once at startup
    and delegates tensor operations to the typed reference; it never falls back
    to generic ``get``/``put`` for these paths.
    """

    @abstractmethod
    async def put_tensor(self, data: Any, **kwargs: Any) -> Any:
        """Write tensor data; return metadata describing stored indexes."""
        ...

    @abstractmethod
    async def get_data(self, batch_meta: Any, fields: list[str] | None = None) -> Any:
        """Fetch tensor data for a batch described by batch_meta."""
        ...

    @abstractmethod
    async def get_meta(
        self,
        fields: list[str],
        batch_size: int,
        task_name: str = "default",
        sampler: Any = None,
        **sampling_kwargs: Any,
    ) -> Any:
        """Return sampling metadata for a training batch."""
        ...


@runtime_checkable
class ConsumptionBackend(Protocol):
    """Extension protocol for backends that track consumption state.

    Implementing this allows BucketStorage to delegate consumption bookkeeping
    to the backend (e.g. for persistent replay or deduplication).
    When absent, BucketStorage maintains its own in-memory tracking.
    """

    @abstractmethod
    async def mark_consumed(self, task_name: str, global_indexes: list[int]) -> None:
        """Mark indexes as consumed for a given task."""
        ...

    @abstractmethod
    async def reset_consumption(self, task_name: str) -> None:
        """Reset consumption state for a given task."""
        ...

    @abstractmethod
    async def clear(self, global_indexes: list[int]) -> None:
        """Remove records at the given global indexes."""
        ...

    @abstractmethod
    async def get_by_indices(self, indexes: list[int]) -> list[dict[str, Any]]:
        """Fetch records by their global indexes (more efficient than repeated get)."""
        ...


class MemoryBackend:
    """Pure In-Memory Backend - Built-in Default Implementation

    Features:
    - No persistence, data exists only in memory
    - Supports blocking wait for new data
    - Lightweight, suitable for testing and temporary data

    For persistence, use a plugin that implements StorageBackend (e.g. register_backend).
    """

    def __init__(self, bucket_id: int, **kwargs):
        self.bucket_id = bucket_id
        self.buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    async def put(self, record: dict[str, Any]) -> None:
        async with self._condition:
            self.buffer.append(record)
            self._condition.notify_all()

    async def put_batch(self, records: list[dict[str, Any]]) -> None:
        async with self._condition:
            self.buffer.extend(records)
            self._condition.notify_all()

    async def get(self, limit: int, offset: int) -> list[dict[str, Any]]:
        async with self._lock:
            return self.buffer[offset : offset + limit]

    async def get_stream(
        self,
        limit: int,
        offset: int,
        wait: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        current_offset = offset
        remaining = limit

        while remaining > 0:
            async with self._condition:
                total = len(self.buffer)

                if current_offset >= total:
                    if wait:
                        try:
                            if timeout:
                                await asyncio.wait_for(
                                    self._condition.wait(), timeout=timeout
                                )
                            else:
                                await self._condition.wait()
                            continue
                        except asyncio.TimeoutError:
                            return
                    else:
                        return

                records = self.buffer[
                    current_offset : current_offset + min(remaining, 100)
                ]

            if records:
                yield records
                current_offset += len(records)
                remaining -= len(records)
            elif not wait:
                break

    async def flush(self) -> None:
        pass  # Pure in-memory, no flush needed

    async def stats(self) -> dict[str, Any]:
        return {
            "bucket_id": self.bucket_id,
            "buffer_size": len(self.buffer),
            "persisted_count": 0,
            "total_count": len(self.buffer),
            "backend": "memory",
        }

    def total_count(self) -> int:
        return len(self.buffer)

    async def put_tensor(self, data: Any, **kwargs: Any) -> Any:
        if isinstance(data, list):
            await self.put_batch(data)
            return {"size": len(data)}
        if isinstance(data, dict):
            await self.put(data)
            return {"size": 1}
        raise TypeError("MemoryBackend.put_tensor expects dict or list[dict]")

    async def get_data(self, batch_meta: Any, fields: list[str] | None = None) -> Any:
        if isinstance(batch_meta, dict):
            indexes = batch_meta.get("global_indexes", [])
        else:
            indexes = getattr(batch_meta, "global_indexes", [])
        rows = [self.buffer[i] for i in indexes if 0 <= i < len(self.buffer)]
        if not fields:
            return rows
        return [{k: v for k, v in row.items() if k in fields} for row in rows]

    async def get_meta(
        self,
        fields: list[str],
        batch_size: int,
        task_name: str = "default",
        sampler: Any = None,
        **sampling_kwargs: Any,
    ) -> Any:
        total = len(self.buffer)
        ready = list(range(total))
        if sampler is not None:
            sampled, _ = sampler.sample(ready, batch_size, **sampling_kwargs)
        else:
            sampled = ready[:batch_size]
        return build_batch_meta(
            sampled, fields, sampling_kwargs.get("partition_id", "default")
        )

    # ---- ConsumptionBackend methods ----

    async def get_by_indices(self, indexes: list[int]) -> list[dict[str, Any]]:
        return [self.buffer[i] for i in indexes if 0 <= i < len(self.buffer)]

    async def mark_consumed(self, task_name: str, global_indexes: list[int]) -> None:
        pass  # BucketStorage owns the consumption state for MemoryBackend

    async def reset_consumption(self, task_name: str) -> None:
        pass

    async def clear(self, global_indexes: list[int]) -> None:
        to_remove = set(global_indexes)
        self.buffer = [r for i, r in enumerate(self.buffer) if i not in to_remove]


# ============================================================
# Backend Registry
# ============================================================

# Built-in backend mapping (only memory)
_BUILTIN_BACKENDS: dict[str, type] = {
    "memory": MemoryBackend,
}

# Plugin backends registered via register_backend()
_REGISTERED_BACKENDS: dict[str, type] = {}


def register_backend(name: str, backend_class: type) -> None:
    """Register a custom storage backend (e.g. from a plugin package).

    Example:
        from my_plugin import MyBackend
        register_backend("my_backend", MyBackend)
        writer = await write_queue(system, "topic", backend="my_backend")
    """
    if not isinstance(backend_class, type):
        raise TypeError(f"backend_class must be a class, got {type(backend_class)}")
    _REGISTERED_BACKENDS[name] = backend_class
    logger.info(f"Registered storage backend: {name}")


def get_backend_class(backend: str | type) -> type:
    """Get backend class

    Args:
        backend: Backend name (str) or backend class (type)

    Returns:
        Backend class
    """
    if isinstance(backend, type):
        return backend

    if backend in _REGISTERED_BACKENDS:
        return _REGISTERED_BACKENDS[backend]

    if backend in _BUILTIN_BACKENDS:
        return _BUILTIN_BACKENDS[backend]

    available = list(_BUILTIN_BACKENDS.keys()) + list(_REGISTERED_BACKENDS.keys())
    raise ValueError(
        f"Unknown backend: {backend}. Available: {available}. "
        "Use register_backend() to add custom backends."
    )


def list_backends() -> list[str]:
    """List all available backends"""
    return list(_BUILTIN_BACKENDS.keys()) + list(_REGISTERED_BACKENDS.keys())
