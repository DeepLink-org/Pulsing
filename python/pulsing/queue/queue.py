"""Queue API - High-level interface based on topic/path"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from pulsing.actor import ActorSystem
from pulsing.actor.remote import ActorProxy

from .manager import get_bucket_ref, get_storage_manager

if TYPE_CHECKING:
    from .sync_queue import SyncQueue, SyncQueueReader, SyncQueueWriter

logger = logging.getLogger(__name__)


class Queue:
    """Distributed Queue - High-level API

    Each bucket corresponds to an independent BucketStorage Actor.

    Args:
        system: Actor system
        topic: Queue topic
        bucket_column: Bucketing column name
        num_buckets: Number of buckets
        batch_size: Batch size
        storage_path: Storage path
        backend: Storage backend
            - "memory": Pure in-memory backend (default)
            - Persistent backend requires installing persisting package
            - Custom class: Class implementing StorageBackend protocol
        backend_options: Additional backend parameters
    """

    def __init__(
        self,
        system: ActorSystem,
        topic: str,
        bucket_column: str = "id",
        num_buckets: int = 4,
        batch_size: int = 100,
        storage_path: str | None = None,
        backend: str | type = "memory",
        backend_options: dict[str, Any] | None = None,
    ):
        self.system = system
        self.topic = topic
        self.bucket_column = bucket_column
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.storage_path = storage_path or f"./queue_storage/{topic}"
        self.backend = backend
        self.backend_options = backend_options

        # Actor proxies for each bucket
        self._bucket_refs: dict[int, ActorProxy] = {}
        self._init_lock = asyncio.Lock()

        # Save event loop reference (for sync wrapper)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _hash_partition(self, value: Any) -> int:
        """Calculate bucket ID based on value"""
        if value is None:
            value = ""
        hash_value = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
        return hash_value % self.num_buckets

    async def _ensure_bucket(self, bucket_id: int) -> ActorProxy:
        """Ensure Actor for specified bucket is created

        Get bucket reference through StorageManager:
        1. Send GetBucket request to local StorageManager
        2. StorageManager uses consistent hashing to determine owner
        3. If this node, create and return; otherwise return redirect
        4. Automatically handle redirects to get bucket on correct node
        """
        if bucket_id in self._bucket_refs:
            return self._bucket_refs[bucket_id]

        async with self._init_lock:
            if bucket_id in self._bucket_refs:
                return self._bucket_refs[bucket_id]

            # Get bucket reference through StorageManager
            self._bucket_refs[bucket_id] = await get_bucket_ref(
                self.system,
                self.topic,
                bucket_id,
                batch_size=self.batch_size,
                storage_path=self.storage_path,
                backend=self.backend,
                backend_options=self.backend_options,
            )
            logger.debug(f"Got bucket {bucket_id} for topic: {self.topic}")
            return self._bucket_refs[bucket_id]

    async def put(
        self, record: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Write data to queue"""
        if isinstance(record, dict):
            records = [record]
            single = True
        elif isinstance(record, list):
            records = record
            single = False
        else:
            raise TypeError("record must be a dict or list of dicts")

        results = []
        for rec in records:
            if self.bucket_column not in rec:
                raise ValueError(f"Missing partition column '{self.bucket_column}'")

            bucket_id = self._hash_partition(rec[self.bucket_column])
            bucket = await self._ensure_bucket(bucket_id)

            # Direct method call via proxy
            await bucket.put(rec)
            results.append({"bucket_id": bucket_id, "status": "ok"})

        return results[0] if single else results

    async def get(
        self,
        bucket_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
        wait: bool = False,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Read data from queue (both in-memory and persistent data visible)"""
        if bucket_id is not None:
            return await self._get_from_bucket(bucket_id, limit, offset, wait, timeout)

        # Read from all buckets
        all_records = []
        for bid in range(self.num_buckets):
            if len(all_records) >= limit:
                break
            records = await self._get_from_bucket(
                bid, limit - len(all_records), offset, wait, timeout
            )
            all_records.extend(records)

        return all_records[:limit]

    async def _get_from_bucket(
        self,
        bucket_id: int,
        limit: int,
        offset: int,
        wait: bool,
        timeout: float | None,
    ) -> list[dict[str, Any]]:
        """Read data from specified bucket"""
        bucket = await self._ensure_bucket(bucket_id)

        # Try streaming read first via proxy
        try:
            records = []
            async for batch in bucket.get_stream(limit, offset, wait, timeout):
                for record in batch:
                    records.append(record)
                    if len(records) >= limit:
                        return records
            return records
        except Exception:
            # Fallback to non-streaming
            return await bucket.get(limit, offset)

    async def flush(self) -> None:
        """Flush all bucket buffers"""
        tasks = []
        for bucket_id in range(self.num_buckets):
            if bucket_id in self._bucket_refs:
                # Direct method call via proxy
                tasks.append(self._bucket_refs[bucket_id].flush())
        if tasks:
            await asyncio.gather(*tasks)

    async def stats(self) -> dict[str, Any]:
        """Get queue statistics"""
        bucket_stats = {}
        for bucket_id in range(self.num_buckets):
            if bucket_id in self._bucket_refs:
                # Direct method call via proxy
                bucket_stats[bucket_id] = await self._bucket_refs[bucket_id].stats()

        return {
            "topic": self.topic,
            "bucket_column": self.bucket_column,
            "num_buckets": self.num_buckets,
            "buckets": bucket_stats,
        }

    def get_bucket_id(self, partition_value: Any) -> int:
        """Calculate bucket ID based on partition column value"""
        return self._hash_partition(partition_value)

    def sync(self) -> "SyncQueue":
        """Return synchronous wrapper

        Example:
            queue = Queue(system, topic="test")
            sync_queue = queue.sync()
            sync_queue.put({"id": "1", "value": 100})  # Synchronous write
            records = sync_queue.get(limit=10)  # Synchronous read
        """
        from .sync_queue import SyncQueue

        return SyncQueue(self)


class QueueWriter:
    """Queue write handle"""

    def __init__(self, queue: Queue):
        self.queue = queue

    async def put(
        self, record: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return await self.queue.put(record)

    async def flush(self) -> None:
        await self.queue.flush()

    def sync(self) -> "SyncQueueWriter":
        """Return synchronous wrapper"""
        from .sync_queue import SyncQueueWriter

        return SyncQueueWriter(self)


class QueueReader:
    """Queue read handle

    Supports three modes:
    1. bucket_ids=None: Read from all buckets
    2. bucket_ids=[0, 2]: Read from specified bucket list
    3. Auto-assign buckets via rank/world_size (distributed consumption)
    """

    def __init__(self, queue: Queue, bucket_ids: list[int] | None = None):
        self.queue = queue
        self.bucket_ids = bucket_ids  # None means read from all buckets
        # Each bucket independently maintains offset
        self._offsets: dict[int, int] = {}

    def _get_bucket_ids(self) -> list[int]:
        """Get list of buckets to read from"""
        if self.bucket_ids is not None:
            return self.bucket_ids
        return list(range(self.queue.num_buckets))

    async def get(
        self,
        limit: int = 100,
        wait: bool = False,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Read data from assigned buckets"""
        bucket_ids = self._get_bucket_ids()
        all_records = []

        for bid in bucket_ids:
            if len(all_records) >= limit:
                break

            offset = self._offsets.get(bid, 0)
            records = await self.queue._get_from_bucket(
                bid, limit - len(all_records), offset, wait, timeout
            )
            self._offsets[bid] = offset + len(records)
            all_records.extend(records)

        return all_records[:limit]

    def reset(self) -> None:
        """Reset offsets for all buckets"""
        self._offsets.clear()

    def set_offset(self, offset: int, bucket_id: int | None = None) -> None:
        """Set offset"""
        if bucket_id is not None:
            self._offsets[bucket_id] = offset
        else:
            for bid in self._get_bucket_ids():
                self._offsets[bid] = offset

    def sync(self) -> "SyncQueueReader":
        """Return synchronous wrapper"""
        from .sync_queue import SyncQueueReader

        return SyncQueueReader(self)


async def write_queue(
    system: ActorSystem,
    topic: str,
    bucket_column: str = "id",
    num_buckets: int = 4,
    batch_size: int = 100,
    storage_path: str | None = None,
    backend: str | type = "memory",
    backend_options: dict[str, Any] | None = None,
) -> QueueWriter:
    """Open queue for writing

    Args:
        system: Actor system
        topic: Queue topic
        bucket_column: Bucketing column name
        num_buckets: Number of buckets
        batch_size: Batch size
        storage_path: Storage path
        backend: Storage backend
            - "memory": Pure in-memory backend (default)
            - Persistent backend requires installing persisting package
            - Custom class: Class implementing StorageBackend protocol
        backend_options: Additional backend parameters

    Example:
        # Use default in-memory backend
        writer = await write_queue(system, "my_queue")

        # Use persisting's Lance backend
        from persisting.queue import LanceBackend
        from pulsing.queue import register_backend
        register_backend("lance", LanceBackend)
        writer = await write_queue(system, "my_queue", backend="lance")
    """
    # Ensure all nodes in cluster have StorageManager
    from .manager import ensure_storage_managers

    await ensure_storage_managers(system)

    queue = Queue(
        system=system,
        topic=topic,
        bucket_column=bucket_column,
        num_buckets=num_buckets,
        batch_size=batch_size,
        storage_path=storage_path,
        backend=backend,
        backend_options=backend_options,
    )
    return QueueWriter(queue)


def _assign_buckets(num_buckets: int, rank: int, world_size: int) -> list[int]:
    """Assign buckets based on rank and world_size

    Uses round-robin allocation strategy to ensure load balancing:
    - world_size=2, num_buckets=4: rank0 -> [0,2], rank1 -> [1,3]
    - world_size=3, num_buckets=4: rank0 -> [0,3], rank1 -> [1], rank2 -> [2]
    """
    return [i for i in range(num_buckets) if i % world_size == rank]


async def read_queue(
    system: ActorSystem,
    topic: str,
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

    Supports three modes:
    1. Default: Read from all buckets
    2. bucket_id/bucket_ids: Read from specified buckets
    3. rank/world_size: Distributed consumption, auto-assign buckets

    Args:
        system: Actor system
        topic: Queue topic
        bucket_id: Single bucket ID (mutually exclusive with bucket_ids)
        bucket_ids: List of bucket IDs (mutually exclusive with bucket_id)
        rank: Current consumer's rank (0 to world_size-1)
        world_size: Total number of consumers
        num_buckets: Number of buckets (for rank/world_size mode)
        storage_path: Storage path
        backend: Storage backend (must match backend used for writing)
        backend_options: Additional backend parameters

    Example:
        # Distributed consumption: 4 buckets, 2 consumers
        reader0 = await read_queue(system, "q", rank=0, world_size=2)  # bucket 0, 2
        reader1 = await read_queue(system, "q", rank=1, world_size=2)  # bucket 1, 3
    """
    # Ensure all nodes in cluster have StorageManager
    from .manager import ensure_storage_managers

    await ensure_storage_managers(system)

    # Determine list of buckets to read from
    if rank is not None and world_size is not None:
        # Distributed consumption mode
        assigned_buckets = _assign_buckets(num_buckets, rank, world_size)
        logger.info(
            f"Reader rank={rank}/{world_size} assigned buckets: {assigned_buckets}"
        )
    elif bucket_id is not None:
        assigned_buckets = [bucket_id]
    elif bucket_ids is not None:
        assigned_buckets = bucket_ids
    else:
        assigned_buckets = None  # Read from all buckets

    # Create Queue (reader side doesn't need bucket_column for hashing, but it must
    # keep `num_buckets/storage_path/backend` consistent with writer).
    queue = Queue(
        system=system,
        topic=topic,
        num_buckets=num_buckets,
        storage_path=storage_path,
        backend=backend,
        backend_options=backend_options,
    )

    # Try to resolve existing bucket Actors
    if assigned_buckets:
        from .storage import BucketStorage

        for bid in assigned_buckets:
            # Must match `StorageManager` bucket actor naming: "bucket_{topic}_{bucket_id}"
            actor_name = f"bucket_{topic}_{bid}"
            try:
                # Use BucketStorage.resolve to get typed ActorProxy
                queue._bucket_refs[bid] = await BucketStorage.resolve(
                    actor_name, system=system
                )
            except Exception:
                pass

    return QueueReader(queue, bucket_ids=assigned_buckets)
