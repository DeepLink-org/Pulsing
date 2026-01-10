# Pulsing Topic - 轻量级 Pub/Sub 模块

## 概述

Topic 模块提供轻量级的 Pub/Sub（发布/订阅）功能，**复用 `queue/manager` 的 StorageManager 进行一致性哈希和集群路由**，确保每个 topic 在集群中只有一个 broker。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                   StorageManager                        │
│  (queue/manager.py - 每节点一个实例)                     │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  GetBucket 处理  │    │      GetTopic 处理          │ │
│  │  (队列 bucket)   │    │      (topic broker)         │ │
│  └─────────────────┘    └─────────────────────────────┘ │
│                                                         │
│  一致性哈希 → 确定 owner 节点 → 创建/返回 ActorRef       │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
   ┌───────────────┐                ┌───────────────┐
   │ BucketStorage │                │  TopicBroker  │
   │  (queue 存储)  │                │  (pub/sub)   │
   └───────────────┘                └───────────────┘
```

## 使用方式

### 发布消息

```python
from pulsing.topic import write_topic

writer = await write_topic(system, "events")
await writer.publish({"type": "user_login", "user_id": 123})
```

### 订阅消息

```python
from pulsing.topic import read_topic

reader = await read_topic(system, "events")

@reader.on_message
async def handle(msg):
    print(f"Received: {msg}")

await reader.start()

# 停止订阅
await reader.stop()
```

### 发布模式

```python
from pulsing.topic import write_topic, PublishMode

writer = await write_topic(system, "events")

# 1. Fire-and-forget（默认）- 发送后立即返回
result = await writer.publish(data)

# 2. Wait all acks - 等待所有订阅者响应
result = await writer.publish(data, mode=PublishMode.WAIT_ALL_ACKS)

# 3. Wait any ack - 等待任一订阅者响应
result = await writer.publish(data, mode=PublishMode.WAIT_ANY_ACK)

# 4. Best effort - 尝试发送，记录失败
result = await writer.publish(data, mode=PublishMode.BEST_EFFORT)
```

## 与 Queue 的关系

| 特性 | Queue | Topic |
|------|-------|-------|
| 消息模式 | 点对点（生产者-消费者） | 广播（发布-订阅） |
| 消息存储 | 持久化（可配置后端） | 无持久化（内存） |
| 消费语义 | 每条消息只被消费一次 | 每条消息被所有订阅者消费 |
| 管理方式 | StorageManager | StorageManager（复用） |

## 公开 API

模块只导出以下必要的 API：

```python
from pulsing.topic import (
    write_topic,      # 获取写入句柄
    read_topic,       # 获取读取句柄
    TopicWriter,      # 写入句柄类型
    TopicReader,      # 读取句柄类型
    PublishMode,      # 发布模式枚举
    PublishResult,    # 发布结果
)
```

## 内部实现

- `TopicBroker`: Broker Actor，管理订阅者和消息分发
- `_SubscriberActor`: 订阅者 Actor，接收消息并调用用户回调
- `StorageManager.GetTopic`: 处理 topic broker 的创建和路由
