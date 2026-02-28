"""Streaming - Queue (point-to-point) and Pub/Sub (topic) APIs

Queue:
    writer = await system.streaming.write("my_queue")  # or system.queue
    reader = await system.streaming.read("my_queue")

Topic:
    writer = await system.topic.write("events")
    reader = await system.topic.read("events")
"""

from typing import TYPE_CHECKING

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
from .queue import Queue, QueueReader, read_queue, write_queue
from .storage import BucketStorage
from .sync_queue import SyncQueue, SyncQueueReader

if TYPE_CHECKING:
    from pulsing._core import ActorSystem


class QueueAPI:
    """Queue API entry point via system.queue — delegates to write_queue/read_queue."""

    def __init__(self, system: "ActorSystem"):
        self._system = system

    async def write(self, topic: str, **kwargs) -> Queue:
        return await write_queue(self._system, topic, **kwargs)

    async def read(self, topic: str, **kwargs) -> QueueReader:
        return await read_queue(self._system, topic, **kwargs)


class TopicAPI:
    """Topic API entry point via system.topic — delegates to write_topic/read_topic."""

    def __init__(self, system: "ActorSystem"):
        self._system = system

    async def write(self, topic: str, **kwargs) -> TopicWriter:
        return await write_topic(self._system, topic, **kwargs)

    async def read(self, topic: str, **kwargs) -> TopicReader:
        return await read_topic(self._system, topic, **kwargs)


__all__ = [
    "QueueAPI",
    "TopicAPI",
    "Queue",
    "QueueReader",
    "write_queue",
    "read_queue",
    "SyncQueue",
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
