"""Storage Backend Protocol - Pluggable Storage Implementation

Defines StorageBackend protocol, allowing different storage implementations:
- MemoryBackend: Pure in-memory (built-in default)
- Third-party implementations: e.g., LanceBackend, PersistingBackend provided by persisting

Usage:
    # Use built-in backend
    writer = await write_queue(system, "topic", backend="memory")

    # Use persistent backend provided by persisting
    from persisting.queue import LanceBackend
    from pulsing.queue import register_backend
    register_backend("lance", LanceBackend)
    writer = await write_queue(system, "topic", backend="lance")

    # Or pass class directly
    writer = await write_queue(system, "topic", backend=LanceBackend)
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Any, AsyncIterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageBackend(Protocol):
    """Storage Backend Protocol

    All storage backends must implement this protocol. Can be implemented via inheritance or duck typing.
    """

    @abstractmethod
    async def put(self, record: dict[str, Any]) -> None:
        """Write a single record"""
        ...

    @abstractmethod
    async def put_batch(self, records: list[dict[str, Any]]) -> None:
        """Write records in batch"""
        ...

    @abstractmethod
    async def get(self, limit: int, offset: int) -> list[dict[str, Any]]:
        """Read records"""
        ...

    @abstractmethod
    async def get_stream(
        self,
        limit: int,
        offset: int,
        wait: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Stream read records"""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Flush buffer to persistent storage"""
        ...

    @abstractmethod
    async def stats(self) -> dict[str, Any]:
        """Get statistics"""
        ...

    @abstractmethod
    def total_count(self) -> int:
        """Total record count"""
        ...


class MemoryBackend:
    """Pure In-Memory Backend - Built-in Default Implementation

    Features:
    - No persistence, data exists only in memory
    - Supports blocking wait for new data
    - Lightweight, suitable for testing and temporary data

    For persistence capabilities, use backends provided by the persisting package:
    - persisting.queue.LanceBackend: Lance persistence
    - persisting.queue.PersistingBackend: Enhanced version (WAL, monitoring, etc.)
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


# ============================================================
# Backend Registry
# ============================================================

# Built-in backend mapping (only memory)
_BUILTIN_BACKENDS: dict[str, type] = {
    "memory": MemoryBackend,
}

# Third-party backend registration (e.g., lance provided by persisting)
_REGISTERED_BACKENDS: dict[str, type] = {}


def register_backend(name: str, backend_class: type) -> None:
    """Register a custom backend

    Example:
        from persisting.queue import LanceBackend
        register_backend("lance", LanceBackend)

        writer = await write_queue(system, "topic", backend="lance")
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
        f"Use register_backend() to add custom backends, or install 'persisting' for Lance support."
    )


def list_backends() -> list[str]:
    """List all available backends"""
    return list(_BUILTIN_BACKENDS.keys()) + list(_REGISTERED_BACKENDS.keys())
