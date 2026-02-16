# 分布式内存队列

Pulsing 内置了一个**分布式内存队列**（Distributed Memory Queue），复用 Pulsing 的 Actor + Cluster 基础设施实现。

它适合：

- **高吞吐写入**（分桶/并行）
- **位置透明访问**（读写端无需知道数据在哪个节点）
- **可插拔存储后端**，内置内存后端，可通过 [Persisting](https://github.com/DeepLink-org/Persisting) 实现持久化

## 架构

- **Topic**：队列主题，例如 `my_queue`
- **Bucket**：一个 topic 被切分为 \(N\) 个 bucket（`num_buckets`）
- **BucketStorage（Actor）**：每个 bucket 对应一个 `BucketStorage` Actor，内部包含：
  - 可插拔的 `StorageBackend` 实例（默认：`MemoryBackend`）
  - 支持通过 `backend` 参数使用自定义后端
- **StorageManager（Actor）**：每个节点一个（服务名 `queue_storage_manager`）
  - 使用**一致性哈希**决定某个 bucket 的 owner 节点
  - 若 bucket 属于本节点：创建/返回本地 `BucketStorage`
  - 否则：返回 Redirect，客户端自动跳转到 owner 节点

### 存储后端

| 后端 | 位置 | 持久化 | 说明 |
|------|------|--------|------|
| `MemoryBackend` | Pulsing（内置） | 否 | 快速内存存储，默认 |
| `LanceBackend` | [Persisting](https://github.com/DeepLink-org/Persisting) | 是 | Lance 列式存储 |
| `PersistingBackend` | [Persisting](https://github.com/DeepLink-org/Persisting) | 是 | 增强版，支持 WAL、指标 |

### 一致性哈希与重定向流程

`StorageManager` 会为每个 `(topic, bucket_id)` 计算 owner 节点，并返回：

- `BucketReady`（bucket 属于本节点）→ 直接使用返回的 `BucketStorage` actor
- `Redirect`（bucket 属于其它节点）→ resolve 远端 `StorageManager` 并重试

```mermaid
flowchart TB
    C[Client: get_bucket_ref(topic, bucket_id)] --> SM[本地 StorageManager]
    SM --> H[一致性哈希计算 owner]
    H --> D{owner == 本地?}
    D -->|是| BR[BucketReady(actor_id, node_id_hex)]
    D -->|否| RD[Redirect(owner_node_id_hex, owner_addr)]

    RD --> RSM[Resolve 远端 StorageManager]
    RSM --> SM2[远端 StorageManager]
    SM2 --> BR2[BucketReady(actor_id, node_id_hex)]

    BR --> REF[ActorSystem.actor_ref(ActorId)]
    BR2 --> REF

    style SM fill:#e3f2fd,stroke:#1976d2
    style SM2 fill:#e3f2fd,stroke:#1976d2
    style RD fill:#fff3e0,stroke:#f57c00
    style BR fill:#e8f5e9,stroke:#388e3c
    style BR2 fill:#e8f5e9,stroke:#388e3c
```

## 快速开始（异步）

```python
import asyncio
import pulsing as pul


async def main():
    await pul.init()
    try:
        writer = await pul.queue.write(
            "my_queue",
            bucket_column="user_id",
            num_buckets=4,
            batch_size=10,
        )
        reader = await pul.queue.read("my_queue")

        # 写入
        await writer.put({"user_id": "u1", "payload": "hello"})

        # 读取（内存 + 已持久化数据同时可见）
        records = await reader.get(limit=10)
        print(records)

        # 持久化缓冲区
        await writer.flush()
    finally:
        await pul.shutdown()


asyncio.run(main())
```

## 同步包装器

如果你需要阻塞式 API（例如在线程里调用），用 `.sync()`：

```python
writer = (await pul.queue.write("my_queue")).sync()
reader = (await pul.queue.read("my_queue")).sync()

writer.put({"id": "1", "value": 100})
records = reader.get(limit=10)
writer.flush()
```

注意：不要在 async 函数内部使用同步包装器（会阻塞事件循环）。

## 分区与分桶

- record **必须**包含 `bucket_column`（默认 `id`）
- bucket 计算为 `md5(str(value)) % num_buckets`
- 同一个 key 永远落到同一个 bucket（稳定分片）

## 读取模式

`pul.queue.read()` 支持：

- **读取所有 bucket**（默认）
- **读取指定 bucket**：`bucket_id=` / `bucket_ids=`
- **分布式消费**：`rank=` / `world_size=` 按轮询分配 bucket

例子：

```python
reader0 = await pul.queue.read("q", rank=0, world_size=2, num_buckets=4)  # [0, 2]
reader1 = await pul.queue.read("q", rank=1, world_size=2, num_buckets=4)  # [1, 3]
```

## 流式读取与阻塞等待

Bucket 默认走流式读取（`GetStream`）：

- **wait=false**：没数据就立刻返回
- **wait=true**：阻塞等待新数据（可选 `timeout`）

## 可见性语义（buffer vs persisted）

每个 `BucketStorage` 内部有两段数据（使用持久化后端时）：

- **持久化段**：由后端存储（如 Lance 数据集）
- **内存缓冲段**：刚写入但尚未 flush 的记录

读者看到的是一个统一的逻辑视图：

```mermaid
flowchart LR
    P[持久化: 0..persisted_count) --> V[按 offset 形成统一视图]
    B[缓冲: persisted_count..total_count) --> V

    style P fill:#e8f5e9,stroke:#388e3c
    style B fill:#fff3e0,stroke:#f57c00
```

**保证**

- `put` 成功后，数据会**立即对读者可见**（至少在内存缓冲中可读到）。

**不保证**

- 若没有使用持久化后端，或 `flush()` 失败，则不保证落盘持久化。

## 存储后端

### 内存后端（默认）

默认的 `MemoryBackend` 将数据存储在内存中，无持久化：

```python
writer = await pul.queue.write(
    "my_queue",
    backend="memory",  # 默认，可省略
)
```

### 使用 Persisting 实现持久化

如需持久化存储，使用 [Persisting](https://github.com/DeepLink-org/Persisting) 提供的后端：

```python
import pulsing as pul
from pulsing.streaming import register_backend
import persisting as pst

# 从 Persisting 注册后端
register_backend("lance", pst.queue.LanceBackend)
register_backend("persisting", pst.queue.PersistingBackend)

await pul.init()

# 使用 Lance 后端实现持久化
writer = await pul.queue.write(
    "my_queue",
    backend="lance",
    storage_path="/data/queues",
)

# 或使用增强版 Persisting 后端（支持 WAL）
writer = await pul.queue.write(
    "my_queue",
    backend="persisting",
    storage_path="/data/queues",
    backend_options={"enable_wal": True},
)
```

### 自定义后端

实现 `StorageBackend` 协议并注册：

```python
from pulsing.streaming import register_backend

class MyBackend:
    async def put(self, record): ...
    async def get(self, offset, limit): ...
    async def flush(self): ...
    # ... 其他方法

register_backend("my_backend", MyBackend)
writer = await pul.queue.write("topic", backend="my_backend")
```

## 多消费者 offset：策略与局限

### offset 如何工作

- 读取是**按 offset** 的（`offset` / `limit`），且按 bucket 分开。
- `QueueReader` 在客户端侧维护**每个 bucket 的 offset**，每次读取后按返回条数推进 offset。

这使得队列更像一个**可重复读取的日志**，而不是“读了就删除”的 destructive queue。

### 分布式消费（`rank` / `world_size`）

传入 `rank/world_size` 时，会按轮询分配 bucket：

- `num_buckets=4, world_size=2`：rank0 → `[0,2]`，rank1 → `[1,3]`

这样可以从结构上避免多个消费者读取同一批 bucket（降低重复消费概率）。

### 局限（重要）

- 没有内建的 **consumer group / ack / commit log**。
- 如果多个消费者独立读取同一个 bucket（各自 offset），可能读到相同记录（重复）。
- offset 默认是客户端内存态；除非你自己持久化，否则重启会丢失进度。

推荐做法：

- 记录里带 **幂等键**（idempotency key），消费端去重。
- 用 actor 状态（或独立 commit log）实现 ack/提交。

## 代码入口

- `python/pulsing/queue/queue.py`：高层 `Queue` / `write_queue` / `read_queue`
- `python/pulsing/queue/manager.py`：`StorageManager`（bucket 路由 / redirect）
- `python/pulsing/queue/storage.py`：`BucketStorage`（委托给 `StorageBackend`）
- `python/pulsing/queue/backend.py`：`StorageBackend` 协议和 `MemoryBackend`
- `examples/python/distributed_queue.py`：端到端示例
- `tests/python/test_queue.py`：行为与压力测试

## 相关项目

- **[Persisting](https://github.com/DeepLink-org/Persisting)**：持久化存储后端（Lance、WAL、指标）
