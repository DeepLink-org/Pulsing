"""分布式内存队列 - 基于 Pulsing Actor 架构

架构特点：
- 每个节点有一个 StorageManager Actor，负责管理本节点的所有 bucket
- StorageManager 使用一致性哈希确定 bucket 的 owner 节点
- 保证整个集群中每个 bucket 只有一个 Actor
- 支持可插拔的存储后端

存储后端：
- "memory": 纯内存后端（内置默认）
- 持久化后端需要安装 persisting 包

Example:
    # 使用默认内存后端
    writer = await write_queue(system, "my_queue")
    
    # 使用 persisting 提供的 Lance 持久化后端
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
    # 后端相关
    "StorageBackend",
    "MemoryBackend",
    "register_backend",
    "get_backend_class",
    "list_backends",
]
