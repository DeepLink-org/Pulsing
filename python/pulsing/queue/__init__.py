"""Distributed In-Memory Queue - Based on Pulsing Actor Architecture

Architecture Features:
- Each node has a StorageManager Actor that manages all buckets on this node
- StorageManager uses consistent hashing to determine the owner node for each bucket
- Ensures only one Actor per bucket across the entire cluster
- Supports pluggable storage backends

Storage Backends:
- "memory": Pure in-memory backend (built-in default)
- Persistent backends require installing the persisting package

Example:
    system = await pul.actor_system()

    # Write to queue
    writer = await system.queue.write("my_queue")
    await writer.put({"id": "1", "data": "hello"})

    # Read from queue
    reader = await system.queue.read("my_queue")
    records = await reader.get(limit=10)
"""

from typing import TYPE_CHECKING, Any

from .backend import (
    MemoryBackend,
    StorageBackend,
    get_backend_class,
    list_backends,
    register_backend,
)
from .manager import (
    StorageManager,
    get_bucket_ref,
    get_storage_manager,
    get_topic_broker,
)
from .queue import Queue, QueueReader, QueueWriter, read_queue, write_queue
from .storage import BucketStorage
from .sync_queue import SyncQueue, SyncQueueReader, SyncQueueWriter

if TYPE_CHECKING:
    from pulsing._core import ActorSystem


class QueueAPI:
    """Queue API entry point via system.queue

    Example:
        system = await pul.actor_system()

        # Write
        writer = await system.queue.write("my_queue")
        await writer.put({"id": "1", "data": "hello"})

        # Read
        reader = await system.queue.read("my_queue")
        records = await reader.get(limit=10)
    """

    def __init__(self, system: "ActorSystem"):
        self._system = system

    async def write(
        self,
        topic: str,
        *,
        bucket_column: str = "id",
        num_buckets: int = 4,
        batch_size: int = 100,
        storage_path: str | None = None,
        backend: str | type = "memory",
        backend_options: dict[str, Any] | None = None,
    ) -> QueueWriter:
        """Open queue for writing

        Args:
            topic: Queue topic name
            bucket_column: Column used for bucketing (default: "id")
            num_buckets: Number of buckets (default: 4)
            batch_size: Batch size for writes (default: 100)
            storage_path: Storage path (default: ./queue_storage/{topic})
            backend: Storage backend ("memory" or custom)
            backend_options: Additional backend options

        Returns:
            QueueWriter for put/flush operations
        """
        return await write_queue(
            self._system,
            topic,
            bucket_column=bucket_column,
            num_buckets=num_buckets,
            batch_size=batch_size,
            storage_path=storage_path,
            backend=backend,
            backend_options=backend_options,
        )

    async def read(
        self,
        topic: str,
        *,
        bucket_id: int | None = None,
        bucket_ids: list[int] | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        num_buckets: int = 4,
        storage_path: str | None = None,
        backend: str | type = "memory",
        backend_options: dict[str, Any] | None = None,
    ) -> QueueReader:
        """Open queue for reading

        Args:
            topic: Queue topic name
            bucket_id: Single bucket to read from
            bucket_ids: List of buckets to read from
            rank: Consumer rank for distributed consumption
            world_size: Total consumers for distributed consumption
            num_buckets: Number of buckets (default: 4)
            storage_path: Storage path
            backend: Storage backend (must match writer)
            backend_options: Additional backend options

        Returns:
            QueueReader for get operations
        """
        return await read_queue(
            self._system,
            topic,
            bucket_id=bucket_id,
            bucket_ids=bucket_ids,
            rank=rank,
            world_size=world_size,
            num_buckets=num_buckets,
            storage_path=storage_path,
            backend=backend,
            backend_options=backend_options,
        )


__all__ = [
    # High-level API
    "QueueAPI",
    # Async API
    "Queue",
    "QueueWriter",
    "QueueReader",
    "write_queue",
    "read_queue",
    # Sync wrapper (obtained via .sync())
    "SyncQueue",
    "SyncQueueWriter",
    "SyncQueueReader",
    # Low-level components
    "StorageManager",
    "BucketStorage",
    "get_storage_manager",
    "get_bucket_ref",
    "get_topic_broker",
    # Backend related
    "StorageBackend",
    "MemoryBackend",
    "register_backend",
    "get_backend_class",
    "list_backends",
]
