"""Streaming - Queue (point-to-point) and Pub/Sub (topic) APIs

Queue:
    writer = await system.streaming.write("my_queue")  # or system.queue
    reader = await system.streaming.read("my_queue")

Topic:
    writer = await system.topic.write("events")
    reader = await system.topic.read("events")
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
from .pubsub import (
    PublishMode,
    PublishResult,
    TopicReader,
    TopicWriter,
    read_topic,
    subscribe_to_topic,
    write_topic,
)
from .queue import Queue, QueueReader, QueueWriter, read_queue, write_queue
from .storage import BucketStorage
from .sync_queue import SyncQueue, SyncQueueReader, SyncQueueWriter

if TYPE_CHECKING:
    from pulsing._core import ActorSystem


class QueueAPI:
    """Queue API entry point via system.queue"""

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
        """Open queue for writing"""
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
        """Open queue for reading"""
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


class TopicAPI:
    """Topic API entry point via system.topic"""

    def __init__(self, system: "ActorSystem"):
        self._system = system

    async def write(
        self,
        topic: str,
        *,
        writer_id: str | None = None,
    ) -> TopicWriter:
        """Open topic for writing"""
        return await write_topic(self._system, topic, writer_id=writer_id)

    async def read(
        self,
        topic: str,
        *,
        reader_id: str | None = None,
        auto_start: bool = False,
    ) -> TopicReader:
        """Open topic for reading"""
        return await read_topic(
            self._system, topic, reader_id=reader_id, auto_start=auto_start
        )


__all__ = [
    "QueueAPI",
    "TopicAPI",
    "Queue",
    "QueueWriter",
    "QueueReader",
    "write_queue",
    "read_queue",
    "SyncQueue",
    "SyncQueueWriter",
    "SyncQueueReader",
    "StorageManager",
    "BucketStorage",
    "get_storage_manager",
    "get_bucket_ref",
    "get_topic_broker",
    "StorageBackend",
    "MemoryBackend",
    "register_backend",
    "get_backend_class",
    "list_backends",
    "write_topic",
    "read_topic",
    "subscribe_to_topic",
    "TopicWriter",
    "TopicReader",
    "PublishMode",
    "PublishResult",
]
