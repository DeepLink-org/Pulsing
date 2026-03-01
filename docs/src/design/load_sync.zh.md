# Pulsing 负载信息同步设计

## 概述

Pulsing Actor System 使用基于 **StreamMessage 的流式订阅** 机制实现 Worker 负载信息的实时同步。

## 架构

```mermaid
sequenceDiagram
    participant R as Router
    participant W as Worker

    R->>W: SubscribeLoad
    W-->>R: StreamMessage {load: 5}
    W-->>R: StreamMessage {load: 3}
    W-->>R: StreamMessage {load: 7}

    Note over R,W: 连接断开 = 自动清理
```

## 核心组件

### Worker 端 (`TransformersWorker`)

```python
class TransformersWorker(Actor):
    async def receive(self, msg):
        if msg.msg_type == "SubscribeLoad":
            return self._handle_subscribe_load()
        # ...

    def _handle_subscribe_load(self) -> StreamMessage:
        """返回持续推送负载的流"""
        stream_msg, writer = StreamMessage.create("LoadStream")
        self._load_subscribers.append(writer)

        async def produce():
            while True:
                await writer.write(self._get_load_snapshot())
                await asyncio.sleep(1.0)

        asyncio.create_task(produce())
        return stream_msg
```

### Router 端 (`StreamLoadScheduler`)

```python
scheduler = StreamLoadScheduler(actor_system, "worker")
await scheduler.start()  # 发现 Worker + 订阅负载流

worker_ref = await scheduler.select_worker()  # 选择负载最低的
```

## 使用方式

### 启动 Router (默认使用 StreamLoadScheduler)

```python
from pulsing.serving import start_router

runner = await start_router(
    system,
    http_port=8080,
    model_name="my-model",
    scheduler_type="stream_load",  # 默认值；也支持 scheduler=... 实例。不支持 scheduler_class。
)
```

### 可选的调度器类型

| 类型 | 说明 | 负载感知 |
|------|------|----------|
| `stream_load` | 流式负载订阅 (默认) | ✅ 实时 |
| `random` | 随机选择 | ❌ |
| `round_robin` | 轮询选择 | ❌ |
| `power_of_two` | Power-of-Two (Rust) | ✅ 本地计数 |
| `cache_aware` | 缓存感知 (Rust) | ✅ 本地计数 |

## 信息流

```mermaid
flowchart TB
    subgraph Step1["1. Router 启动"]
        A1[StreamLoadScheduler.start] --> A2[Gossip 发现所有 Worker]
        A2 --> A3[向每个 Worker 发送 SubscribeLoad]
    end

    subgraph Step2["2. Worker 返回 StreamMessage"]
        B1[启动异步任务] --> B2[持续推送负载]
    end

    subgraph Step3["3. Router 接收流"]
        C1[更新本地负载缓存]
    end

    subgraph Step4["4. 路由请求"]
        D1[scheduler.select_worker] --> D2[返回负载最低的 Worker]
    end

    subgraph Step5["5. Worker 下线"]
        E1[Stream 断开] --> E2[Router 自动清理]
    end

    Step1 --> Step2 --> Step3 --> Step4
    Step3 -.-> Step5

    style Step1 fill:#e3f2fd,stroke:#1976d2
    style Step2 fill:#fff3e0,stroke:#f57c00
    style Step3 fill:#e8f5e9,stroke:#388e3c
    style Step4 fill:#f3e5f5,stroke:#7b1fa2
    style Step5 fill:#ffebee,stroke:#c62828
```

## 与 Gossip 的配合

| 机制 | 职责 |
|------|------|
| Gossip | 服务发现、成员管理 |
| StreamMessage | 实时状态同步 |

```python
# Gossip: 发现 Worker
workers = await gossip.get_named_actor_instances("worker")

# StreamMessage: 订阅负载
for worker in workers:
    await consumer.subscribe(worker)
```
