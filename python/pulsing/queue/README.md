# 分布式内存队列

基于 Pulsing Actor 架构实现的分布式内存队列系统。

**支持可插拔存储后端**，可根据需求选择不同的实现。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              应用层                                          │
│                                                                             │
│     Queue / QueueWriter / QueueReader                                       │
│       │                                                                     │
│       │  get_bucket_ref(topic, bucket_id)                                   │
│       ▼                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    StorageManager (每节点一个)                               │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  GetBucket(topic, bucket_id)                                        │   │
│   │       │                                                             │   │
│   │       ├─ owner = hash(topic:bucket_id) % nodes  ← 一致性哈希        │   │
│   │       │                                                             │   │
│   │       ├─ owner == self?                                             │   │
│   │       │      ├─ Yes → 创建/返回 BucketStorage                       │   │
│   │       │      │        → BucketReady(actor_id, node_id)              │   │
│   │       │      │                                                      │   │
│   │       │      └─ No  → Redirect(owner_node_id)                       │   │
│   │       │              客户端重定向到正确节点                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│BucketStorage │           │BucketStorage │           │BucketStorage │
│   bucket_0   │           │   bucket_1   │           │   bucket_2   │
│              │           │              │           │              │
│  - buffer[]  │           │  - buffer[]  │           │  - buffer[]  │
│  - Lance     │           │  - Lance     │           │  - Lance     │
│  - Condition │           │  - Condition │           │  - Condition │
└──────────────┘           └──────────────┘           └──────────────┘
     Node A                     Node B                     Node A
```

## 核心组件

### 1. StorageManager（存储管理器）

**每个节点一个实例**，负责管理两类资源：

| 资源类型 | 请求消息 | Actor 类型 | 用途 |
|---------|---------|-----------|------|
| Queue Bucket | `GetBucket` | `BucketStorage` | 生产者-消费者队列 |
| Topic Broker | `GetTopic` | `TopicBroker` | 发布-订阅 |

核心职责：
- 使用**一致性哈希**判断资源的 owner 节点
- Owner 节点：创建并返回对应 Actor
- 非 Owner 节点：返回 `Redirect`，指向正确节点

### 2. BucketStorage（桶存储）

每个 bucket 一个实例，负责：
- 数据缓冲（内存）
- 数据持久化（Lance）
- 消费者阻塞/唤醒（asyncio.Condition）

### 3. Queue / QueueWriter / QueueReader

高级 API，对用户隐藏底层复杂性。

## 设计特点

| 特性 | 说明 |
|------|------|
| **集群唯一性** | StorageManager + 一致性哈希，确保每个 bucket 在集群中只有一个 Actor |
| **智能路由** | 错误请求自动重定向到正确节点 |
| **独立锁/条件变量** | 每个 bucket 独立，无跨 bucket 竞争 |
| **流式传输** | 消费者通过 StreamMessage 接收，内存友好 |
| **实时通知** | 新数据通过 condition + 流推送，无轮询 |

## 请求流程

### Bucket 获取流程

```
Queue.put(record)
    │
    ├─ bucket_id = hash(record[bucket_column]) % num_buckets
    │
    ▼
get_bucket_ref(system, topic, bucket_id)
    │
    ├─ 本地 StorageManager.GetBucket(...)
    │       │
    │       ├─ owner == self?
    │       │      ├─ Yes → BucketReady(actor_id) → 返回 ActorRef
    │       │      └─ No  → Redirect(owner_node_id)
    │       │                    │
    │       │                    ▼
    │       │              owner 节点的 StorageManager.GetBucket(...)
    │       │                    │
    │       │                    └─ BucketReady → 返回 ActorRef
    │
    └─ bucket_ref.ask(Put, {record})
```

### 数据传输流程

```
生产者                    BucketStorage                   消费者 (wait=True)
   │                           │                               │
   │── Put ───────────────────▶│                               │
   │                           │ buffer.append()               │
   │                           │ condition.notify_all() ──────▶│ 唤醒
   │◀── PutResponse ───────────│                               │
   │                           │                               │
   │                           │◀── GetStream ─────────────────│
   │                           │                               │
   │                           │── StreamMessage chunk ───────▶│ 流式发送
```

## 数据可见性模型

```
┌─────────────────────────────────────────────────────┐
│                    总数据视图                        │
├─────────────────────────┬───────────────────────────┤
│    持久化 (Lance)        │      内存缓冲             │
│    [0, persisted_count) │  [persisted_count, total) │
└─────────────────────────┴───────────────────────────┘
                          ↑
                    两部分同时可见
```

- 写入后数据**立即**对消费者可见（在内存缓冲中）
- 达到 `batch_size` 后自动持久化到 Lance
- 调用 `flush()` 可强制持久化

## 快速开始

```python
import asyncio
import pulsing as pul

async def main():
    system = await pul.actor_system()

    # 生产者
    writer = await system.queue.write(
        "my_queue",
        bucket_column="user_id",
        num_buckets=4,
    )

    # 写入数据（立即对消费者可见）
    await writer.put({"user_id": "u1", "message": "Hello"})

    # 消费者
    reader = await system.queue.read("my_queue")

    # 读取数据（内存 + 持久化同时可见）
    records = await reader.get(limit=100)

    # 阻塞等待新数据
    records = await reader.get(limit=100, wait=True, timeout=10.0)

    await system.shutdown()

asyncio.run(main())
```

### 同步 API

通过 `.sync()` 获取同步包装器，可与异步混用：

```python
# 生产者用同步，消费者用异步
sync_writer = writer.sync()
sync_writer.put({"user_id": "u1", "message": "Hello"})
sync_writer.flush()

records = await reader.get(limit=100)  # 异步读取

# 或者反过来
await writer.put({"user_id": "u2", "message": "World"})

sync_reader = reader.sync()
records = sync_reader.get(limit=100)  # 同步读取
```

## API

### `system.queue.write(topic, ...)`

打开队列用于写入。

```python
writer = await system.queue.write(
    "my_queue",
    bucket_column="user_id",  # 分桶列
    num_buckets=4,            # 桶数量
    batch_size=100,           # 批处理大小
)

await writer.put({"user_id": "u1", "msg": "hello"})
await writer.put([record1, record2, ...])  # 批量写入
await writer.flush()  # 强制持久化
```

### `system.queue.read(topic, ...)`

打开队列用于读取。支持三种模式：

```python
# 1. 读取所有 bucket
reader = await system.queue.read("my_queue")

# 2. 读取指定 bucket
reader = await system.queue.read("my_queue", bucket_id=0)
reader = await system.queue.read("my_queue", bucket_ids=[0, 2])

# 3. 分布式消费：通过 rank/world_size 自动分配 bucket
reader0 = await system.queue.read("q", rank=0, world_size=2, num_buckets=4)  # bucket 0, 2
reader1 = await system.queue.read("q", rank=1, world_size=2, num_buckets=4)  # bucket 1, 3

# 读取数据
records = await reader.get(limit=100)
records = await reader.get(limit=100, wait=True, timeout=10.0)  # 阻塞等待
```

## 分布式消费

通过 `rank` 和 `world_size` 实现多消费者并行消费：

```
num_buckets=4, world_size=2:

Consumer (rank=0)           Consumer (rank=1)
      │                           │
      ├─▶ bucket_0                ├─▶ bucket_1
      └─▶ bucket_2                └─▶ bucket_3
```

## 依赖

```bash
pip install lance pyarrow
```

---

## 可插拔存储后端

队列系统支持可插拔的存储后端，可根据需求选择不同实现。

### 内置后端

| 后端 | 说明 | 适用场景 |
|------|------|----------|
| `memory` | 纯内存，无持久化（默认） | 测试、临时数据 |

### 持久化后端（需安装 persisting）

```bash
pip install persisting[lance]
```

| 后端 | 说明 | 适用场景 |
|------|------|----------|
| `LanceBackend` | Lance 持久化 | 一般持久化场景 |
| `PersistingBackend` | 增强版（WAL、监控） | 生产环境 |

### 使用方式

```python
# 使用默认内存后端
writer = await system.queue.write("my_queue")

# 使用 persisting 的 Lance 持久化后端
from persisting.queue import LanceBackend
from pulsing.queue import register_backend

register_backend("lance", LanceBackend)
writer = await system.queue.write("my_queue", backend="lance")

# 使用增强版后端
from persisting.queue import PersistingBackend
register_backend("persisting", PersistingBackend)
writer = await system.queue.write(
    "my_queue",
    backend="persisting",
    backend_options={"enable_wal": True, "enable_metrics": True}
)
```

### 自定义后端

实现 `StorageBackend` 协议即可：

```python
class MyBackend:
    def __init__(self, bucket_id: int, storage_path: str, **kwargs):
        ...

    async def put(self, record: dict) -> None: ...
    async def put_batch(self, records: list[dict]) -> None: ...
    async def get(self, limit: int, offset: int) -> list[dict]: ...
    async def get_stream(self, limit, offset, wait, timeout) -> AsyncIterator: ...
    async def flush(self) -> None: ...
    async def stats(self) -> dict: ...
    def total_count(self) -> int: ...
```

---

## 设计点评

### ✅ 优点

1. **集群唯一性保证**
   - StorageManager 使用一致性哈希确定 bucket owner
   - 非 owner 节点返回 Redirect，避免创建重复 Actor
   - 解决了分布式环境下的竞态条件问题

2. **架构清晰**
   - 三层架构：应用层 (Queue) → 管理层 (StorageManager) → 存储层 (BucketStorage)
   - 职责分离，每层只关注自己的逻辑

3. **智能路由**
   - 客户端无需知道 bucket 在哪个节点
   - 自动重定向到正确节点

4. **高并发支持**
   - 每个 bucket 独立的锁和条件变量
   - 无跨 bucket 竞争

5. **数据实时可见**
   - 写入后立即可读（内存缓冲）
   - 无需等待持久化

6. **流式传输**
   - 大数据量传输内存友好
   - 支持阻塞等待新数据

### ⚠️ 潜在改进点

1. **节点变化处理**
   - 当前一致性哈希在节点加入/退出时可能导致 bucket 重分布
   - 可以考虑虚拟节点或一致性哈希环来减少影响

2. **元数据持久化**
   - Queue 配置（bucket_column, num_buckets）目前不持久化
   - 消费者需要知道这些参数
   - 可以考虑将元数据存储在集群中

3. **故障恢复**
   - 节点故障时，其 bucket 数据可能丢失（内存部分）
   - 可以考虑副本或 WAL 机制

4. **Lance Schema 演化**
   - 当前不支持 schema 变化
   - 新字段可能导致写入失败

5. **性能优化**
   - `get_bucket_ref` 每次都查询 StorageManager
   - 可以增加客户端缓存，减少 RPC 调用

### 📊 适用场景

- ✅ 分布式数据管道
- ✅ 生产者-消费者模式
- ✅ 分布式训练数据分发
- ✅ 实时数据流处理
- ⚠️ 不适合需要强一致性的场景
- ⚠️ 不适合需要事务的场景
