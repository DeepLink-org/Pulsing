"""队列 API - 基于 topic/path 的高级接口"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from pulsing.actor import ActorRef, ActorSystem, Message

from .manager import get_bucket_ref, get_storage_manager

if TYPE_CHECKING:
    from .sync_queue import SyncQueue, SyncQueueReader, SyncQueueWriter

logger = logging.getLogger(__name__)


class Queue:
    """分布式队列 - 高级 API

    每个 bucket 对应一个独立的 BucketStorage Actor。

    Args:
        system: Actor 系统
        topic: 队列主题
        bucket_column: 分桶列名
        num_buckets: 桶数量
        batch_size: 批处理大小
        storage_path: 存储路径
        backend: 存储后端
            - "memory": 纯内存后端（默认）
            - 持久化后端需安装 persisting 包
            - 自定义类: 实现 StorageBackend 协议的类
        backend_options: 后端额外参数
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

        # 每个 bucket 的 Actor 引用
        self._bucket_refs: dict[int, ActorRef] = {}
        self._init_lock = asyncio.Lock()

        # 保存事件循环引用（用于 sync 包装器）
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _hash_partition(self, value: Any) -> int:
        """根据值计算分桶 ID"""
        if value is None:
            value = ""
        hash_value = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
        return hash_value % self.num_buckets

    async def _ensure_bucket(self, bucket_id: int) -> ActorRef:
        """确保指定 bucket 的 Actor 已创建

        通过 StorageManager 获取 bucket 引用：
        1. 向本地 StorageManager 发送 GetBucket 请求
        2. StorageManager 使用一致性哈希判断 owner
        3. 如果是本节点，创建并返回；否则返回重定向
        4. 自动处理重定向，最终获取正确节点上的 bucket
        """
        if bucket_id in self._bucket_refs:
            return self._bucket_refs[bucket_id]

        async with self._init_lock:
            if bucket_id in self._bucket_refs:
                return self._bucket_refs[bucket_id]

            # 通过 StorageManager 获取 bucket 引用
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
        """写入数据到队列"""
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
            bucket_ref = await self._ensure_bucket(bucket_id)

            response = await bucket_ref.ask(Message.from_json("Put", {"record": rec}))
            if response.msg_type == "Error":
                raise RuntimeError(f"Put failed: {response.to_json().get('error')}")

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
        """从队列读取数据（内存+持久化数据同时可见）"""
        if bucket_id is not None:
            return await self._get_from_bucket(bucket_id, limit, offset, wait, timeout)

        # 从所有 bucket 读取
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
        """从指定 bucket 读取数据"""
        bucket_ref = await self._ensure_bucket(bucket_id)

        # 使用流式读取
        response = await bucket_ref.ask(
            Message.from_json(
                "GetStream",
                {"limit": limit, "offset": offset, "wait": wait, "timeout": timeout},
            )
        )

        if response.msg_type == "Error":
            raise RuntimeError(f"Get failed: {response.to_json().get('error')}")

        if not response.is_stream:
            # 回退到非流式
            response = await bucket_ref.ask(
                Message.from_json("Get", {"limit": limit, "offset": offset})
            )
            return response.to_json().get("records", [])

        records = []
        reader = response.stream_reader()
        async for chunk in reader:
            for record in chunk.get("records", []):
                records.append(record)
                if len(records) >= limit:
                    return records

        return records

    async def flush(self) -> None:
        """刷新所有 bucket 的缓冲区"""
        tasks = []
        for bucket_id in range(self.num_buckets):
            if bucket_id in self._bucket_refs:
                tasks.append(
                    self._bucket_refs[bucket_id].ask(Message.from_json("Flush", {}))
                )
        if tasks:
            await asyncio.gather(*tasks)

    async def stats(self) -> dict[str, Any]:
        """获取队列统计信息"""
        bucket_stats = {}
        for bucket_id in range(self.num_buckets):
            if bucket_id in self._bucket_refs:
                response = await self._bucket_refs[bucket_id].ask(
                    Message.from_json("Stats", {})
                )
                bucket_stats[bucket_id] = response.to_json()

        return {
            "topic": self.topic,
            "bucket_column": self.bucket_column,
            "num_buckets": self.num_buckets,
            "buckets": bucket_stats,
        }

    def get_bucket_id(self, partition_value: Any) -> int:
        """根据分桶列的值计算 bucket ID"""
        return self._hash_partition(partition_value)

    def sync(self) -> "SyncQueue":
        """返回同步包装器

        Example:
            queue = Queue(system, topic="test")
            sync_queue = queue.sync()
            sync_queue.put({"id": "1", "value": 100})  # 同步写入
            records = sync_queue.get(limit=10)  # 同步读取
        """
        from .sync_queue import SyncQueue

        return SyncQueue(self)


class QueueWriter:
    """队列写入句柄"""

    def __init__(self, queue: Queue):
        self.queue = queue

    async def put(
        self, record: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return await self.queue.put(record)

    async def flush(self) -> None:
        await self.queue.flush()

    def sync(self) -> "SyncQueueWriter":
        """返回同步包装器"""
        from .sync_queue import SyncQueueWriter

        return SyncQueueWriter(self)


class QueueReader:
    """队列读取句柄

    支持三种模式：
    1. bucket_ids=None: 从所有 bucket 读取
    2. bucket_ids=[0, 2]: 从指定的 bucket 列表读取
    3. 通过 rank/world_size 自动分配 bucket（分布式消费）
    """

    def __init__(self, queue: Queue, bucket_ids: list[int] | None = None):
        self.queue = queue
        self.bucket_ids = bucket_ids  # None 表示读取所有 bucket
        # 每个 bucket 独立维护 offset
        self._offsets: dict[int, int] = {}

    def _get_bucket_ids(self) -> list[int]:
        """获取要读取的 bucket 列表"""
        if self.bucket_ids is not None:
            return self.bucket_ids
        return list(range(self.queue.num_buckets))

    async def get(
        self,
        limit: int = 100,
        wait: bool = False,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """从分配的 bucket 中读取数据"""
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
        """重置所有 bucket 的偏移量"""
        self._offsets.clear()

    def set_offset(self, offset: int, bucket_id: int | None = None) -> None:
        """设置偏移量"""
        if bucket_id is not None:
            self._offsets[bucket_id] = offset
        else:
            for bid in self._get_bucket_ids():
                self._offsets[bid] = offset

    def sync(self) -> "SyncQueueReader":
        """返回同步包装器"""
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
    """打开队列用于写入

    Args:
        system: Actor 系统
        topic: 队列主题
        bucket_column: 分桶列名
        num_buckets: 桶数量
        batch_size: 批处理大小
        storage_path: 存储路径
        backend: 存储后端
            - "memory": 纯内存后端（默认）
            - 持久化后端需安装 persisting 包
            - 自定义类: 实现 StorageBackend 协议的类
        backend_options: 后端额外参数

    Example:
        # 使用默认内存后端
        writer = await write_queue(system, "my_queue")

        # 使用 persisting 的 Lance 后端
        from persisting.queue import LanceBackend
        from pulsing.queue import register_backend
        register_backend("lance", LanceBackend)
        writer = await write_queue(system, "my_queue", backend="lance")
    """
    # 确保集群中所有节点都有 StorageManager
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
    """根据 rank 和 world_size 分配 bucket

    采用轮询分配策略，确保负载均衡：
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
    """打开队列用于读取

    支持三种模式：
    1. 默认：从所有 bucket 读取
    2. bucket_id/bucket_ids：从指定的 bucket 读取
    3. rank/world_size：分布式消费，自动分配 bucket

    Args:
        system: Actor 系统
        topic: 队列主题
        bucket_id: 单个 bucket ID（与 bucket_ids 互斥）
        bucket_ids: bucket ID 列表（与 bucket_id 互斥）
        rank: 当前消费者的 rank（0 到 world_size-1）
        world_size: 消费者总数
        num_buckets: bucket 数量（用于 rank/world_size 模式）
        storage_path: 存储路径
        backend: 存储后端（需与写入时使用的后端一致）
        backend_options: 后端额外参数

    Example:
        # 分布式消费：4 个 bucket，2 个消费者
        reader0 = await read_queue(system, "q", rank=0, world_size=2)  # bucket 0, 2
        reader1 = await read_queue(system, "q", rank=1, world_size=2)  # bucket 1, 3
    """
    # 确保集群中所有节点都有 StorageManager
    from .manager import ensure_storage_managers

    await ensure_storage_managers(system)

    # 确定要读取的 bucket 列表
    if rank is not None and world_size is not None:
        # 分布式消费模式
        assigned_buckets = _assign_buckets(num_buckets, rank, world_size)
        logger.info(
            f"Reader rank={rank}/{world_size} assigned buckets: {assigned_buckets}"
        )
    elif bucket_id is not None:
        assigned_buckets = [bucket_id]
    elif bucket_ids is not None:
        assigned_buckets = bucket_ids
    else:
        assigned_buckets = None  # 读取所有 bucket

    # 创建 Queue
    queue = Queue(
        system=system,
        topic=topic,
        bucket_column="id",
        num_buckets=num_buckets,
        batch_size=100,
        storage_path=storage_path,
        backend=backend,
        backend_options=backend_options,
    )

    # 尝试解析已存在的 bucket Actor
    if assigned_buckets:
        for bid in assigned_buckets:
            actor_name = f"queue_{topic}_bucket_{bid}"
            try:
                queue._bucket_refs[bid] = await system.resolve_named(actor_name)
            except Exception:
                pass

    return QueueReader(queue, bucket_ids=assigned_buckets)
