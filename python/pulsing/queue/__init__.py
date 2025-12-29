"""分布式内存队列 - 基于 Pulsing Actor 架构

架构特点：
- 每个节点有一个 StorageManager Actor，负责管理本节点的所有 bucket
- StorageManager 使用一致性哈希确定 bucket 的 owner 节点
- 保证整个集群中每个 bucket 只有一个 Actor
"""

from .manager import StorageManager, get_bucket_ref, get_storage_manager
from .queue import Queue, QueueReader, QueueWriter, read_queue, write_queue
from .storage import BucketStorage
from .sync_queue import SyncQueue, SyncQueueReader, SyncQueueWriter

__all__ = [
    # 异步 API
    "Queue",
    "QueueWriter",
    "QueueReader",
    "write_queue",
    "read_queue",
    # 同步包装器（通过 .sync() 获取）
    "SyncQueue",
    "SyncQueueWriter",
    "SyncQueueReader",
    # 底层组件
    "StorageManager",
    "BucketStorage",
    "get_storage_manager",
    "get_bucket_ref",
]
