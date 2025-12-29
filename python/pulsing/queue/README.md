# 分布式内存队列

基于 Pulsing Actor 架构实现的分布式内存队列系统。

## 架构概览

```
                           ┌─────────────────────────────────────────┐
                           │              Queue (协调层)              │
                           │  - hash 路由                            │
                           │  - bucket Actor 管理                    │
                           └────────────┬────────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               
   bucket_0        bucket_1        bucket_2        bucket_3        
        │               │               │               │
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│BucketStorage │ │BucketStorage │ │BucketStorage │ │BucketStorage │
│  Actor       │ │  Actor       │ │  Actor       │ │  Actor       │
│              │ │              │ │              │ │              │
│  - buffer[]  │ │  - buffer[]  │ │  - buffer[]  │ │  - buffer[]  │
│  - Lance     │ │  - Lance     │ │  - Lance     │ │  - Lance     │
│  - Condition │ │  - Condition │ │  - Condition │ │  - Condition │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## 设计特点

| 特性 | 说明 |
|------|------|
| **每 bucket 独立 Actor** | 真正分布式，可部署在不同节点 |
| **独立锁和条件变量** | 无跨 bucket 竞争，高并发 |
| **命名 Actor 发现** | `queues/{topic}/bucket_{id}` 支持跨节点发现 |
| **流式传输** | 消费者通过 StreamMessage 接收，内存友好 |
| **实时通知** | 新数据通过 condition + 流推送，无轮询 |

## 数据传输机制

### 流式传输 + 实时通知

```
生产者                    BucketStorage                   消费者 (wait=True)
   │                           │                               │
   │── Put ───────────────────▶│                               │
   │                           │ buffer.append()               │
   │                           │ condition.notify_all() ──────▶│ 唤醒
   │◀── PutResponse ───────────│                               │
   │                           │                               │
   │                           │◀──────────────────────────────│ 继续循环
   │                           │                               │
   │                           │── StreamMessage chunk ───────▶│ 流式发送
```

- **消费者读取**：通过 `StreamMessage` 流式接收，支持大数据量
- **新数据通知**：生产者写入后 `condition.notify_all()` 唤醒等待的流
- **流保持打开**：`wait=True` 时流不关闭，持续等待新数据

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
from pulsing.actor import SystemConfig, create_actor_system
from pulsing.queue import read_queue, write_queue

async def main():
    system = await create_actor_system(SystemConfig.standalone())
    
    # 生产者
    writer = await write_queue(
        system,
        topic="my_queue",
        partition_column="user_id",
        num_buckets=4,
    )
    
    # 写入数据（立即对消费者可见）
    await writer.put({"user_id": "u1", "message": "Hello"})
    
    # 消费者
    reader = await read_queue(system, topic="my_queue")
    
    # 读取数据（内存 + 持久化同时可见）
    records = await reader.get(limit=100)
    
    # 阻塞等待新数据
    records = await reader.get(limit=100, wait=True, timeout=10.0)
    
    await system.shutdown()

asyncio.run(main())
```

## API

### `write_queue(system, topic, ...)`

打开队列用于写入。

```python
writer = await write_queue(
    system,
    topic="my_queue",
    partition_column="user_id",  # 分桶列
    num_buckets=4,               # 桶数量
    batch_size=100,              # 批处理大小
)

await writer.put({"user_id": "u1", "msg": "hello"})
await writer.put([record1, record2, ...])  # 批量写入
await writer.flush()  # 强制持久化
```

### `read_queue(system, topic, ...)`

打开队列用于读取。支持三种模式：

```python
# 1. 读取所有 bucket
reader = await read_queue(system, topic="my_queue")

# 2. 读取指定 bucket
reader = await read_queue(system, topic="my_queue", bucket_id=0)
reader = await read_queue(system, topic="my_queue", bucket_ids=[0, 2])

# 3. 分布式消费：通过 rank/world_size 自动分配 bucket
# 4 个 bucket，2 个消费者
reader0 = await read_queue(system, "q", rank=0, world_size=2, num_buckets=4)  # bucket 0, 2
reader1 = await read_queue(system, "q", rank=1, world_size=2, num_buckets=4)  # bucket 1, 3

# 读取数据
records = await reader.get(limit=100)
records = await reader.get(limit=100, wait=True, timeout=10.0)  # 阻塞等待
```

### `Queue` 类

直接使用 Queue 类进行更细粒度控制：

```python
from pulsing.queue import Queue

queue = Queue(system, topic="my_queue", partition_column="user_id", num_buckets=4)

await queue.put({"user_id": "u1", "msg": "hello"})
records = await queue.get(bucket_id=0, limit=100)
stats = await queue.stats()
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

**优点：**
- 每个消费者只连接部分 bucket，减少连接数
- 负载均衡，数据均匀分配
- 适合分布式训练场景

```python
# 4 个 bucket，2 个消费者并行消费
async def consume(rank: int, world_size: int):
    reader = await read_queue(
        system, "my_queue",
        rank=rank, world_size=world_size, num_buckets=4,
    )
    while True:
        records = await reader.get(limit=100, wait=True)
        process(records)

# 启动 2 个消费者
await asyncio.gather(
    consume(rank=0, world_size=2),  # 处理 bucket 0, 2
    consume(rank=1, world_size=2),  # 处理 bucket 1, 3
)
```

## 依赖

```bash
pip install lance pyarrow
```
