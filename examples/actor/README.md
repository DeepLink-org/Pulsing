# Pulsing Actor System - Python Examples

这个目录包含了如何在 Python 中使用 Pulsing Actor System 的示例。

## 概述

Pulsing Actor System 是一个轻量级的分布式 Actor 框架，具有以下特点：

- **零外部依赖**: 不需要 etcd、nats 或 redis
- **基于 Gossip 的发现**: 使用 SWIM 协议自动发现集群成员
- **位置透明的 ActorRef**: 本地和远程 Actor 使用相同的 API
- **原生异步支持**: 基于 tokio 构建

## 核心概念

### Actor

Actor 是独立的计算单元，通过消息传递进行通信。在 Python 中，你可以通过继承 `Actor` 基类来创建自己的 Actor：

```python
from pulsing.actor import Actor, RawMessage

class MyActor(Actor):
    def __init__(self):
        self.state = {}
    
    def on_start(self, actor_id):
        print(f"Actor {actor_id} started")
    
    async def receive(self, msg: RawMessage) -> RawMessage:
        # 处理消息
        data = msg.to_json()
        return RawMessage.from_json("response", {"result": "ok"})
```

### 消息传递

支持两种消息模式：

1. **Ask 模式**: 发送消息并等待响应
   ```python
   response = await actor_ref.ask_json("message_type", {"key": "value"})
   ```

2. **Tell 模式**: 发送消息，不等待响应（fire-and-forget）
   ```python
   await actor_ref.tell_json("message_type", {"key": "value"})
   ```

### 集群模式

可以通过指定种子节点来加入集群：

```python
config = SystemConfig.with_addr("0.0.0.0:8001").with_seeds([
    "192.168.1.100:8000"
])
system = await ActorSystem.create(config)
```

## 示例

### ping_pong.py

基本的 Actor 通信示例，展示了：
- 创建 ActorSystem
- 定义同步和异步 Actor
- 使用 ask/tell 模式发送消息

运行：
```bash
python examples/actor/ping_pong.py
```

### cluster.py

分布式集群示例，展示了：
- 在不同端口启动多个节点
- 通过种子节点加入集群
- 查看集群成员

运行：
```bash
# 终端 1 - 启动第一个节点
python examples/actor/cluster.py --port 8000

# 终端 2 - 启动第二个节点并加入集群
python examples/actor/cluster.py --port 8001 --seed 127.0.0.1:8000
```

## API 参考

### 类

| 类 | 描述 |
|---|---|
| `ActorSystem` | Actor 系统，管理 Actor 和集群成员 |
| `Actor` | Actor 基类，用户需要继承此类 |
| `ActorRef` | Actor 引用，用于发送消息 |
| `ActorId` | Actor 唯一标识符 |
| `NodeId` | 节点唯一标识符 |
| `RawMessage` | 原始消息，包含类型和负载 |
| `SystemConfig` | 系统配置 |

### ActorSystem 方法

| 方法 | 描述 |
|---|---|
| `create(config)` | 创建新的 Actor 系统（异步） |
| `spawn(name, handler)` | 创建新的 Actor（异步） |
| `actor_ref(actor_id)` | 获取 Actor 引用（异步） |
| `members()` | 获取集群成员（异步） |
| `shutdown()` | 关闭系统（异步） |

### ActorRef 方法

| 方法 | 描述 |
|---|---|
| `ask(msg)` | 发送消息并等待响应（异步） |
| `ask_json(msg_type, data)` | 发送 JSON 消息并等待响应（异步） |
| `tell(msg)` | 发送消息，不等待响应（异步） |
| `tell_json(msg_type, data)` | 发送 JSON 消息，不等待响应（异步） |

### RawMessage 方法

| 方法 | 描述 |
|---|---|
| `from_json(msg_type, data)` | 从 Python 对象创建消息（类方法） |
| `to_json()` | 将消息负载解析为 Python 对象 |
| `empty()` | 创建空响应消息（类方法） |

