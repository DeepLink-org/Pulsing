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
    # Use default in-memory backend
    writer = await write_queue(system, "my_queue")

    # Use Lance persistent backend provided by persisting
    from persisting.queue import LanceBackend
    from pulsing.queue import register_backend
    register_backend("lance", LanceBackend)
    writer = await write_queue(system, "my_queue", backend="lance")
"""

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

__all__ = [
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
