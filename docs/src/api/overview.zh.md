# API 概述

Pulsing 是一个分布式 actor 框架，为构建分布式系统提供通信骨干，并为 AI 应用提供专门支持。

## 核心概念

### Actor 系统

Pulsing 基于[Actor 模型](https://en.wikipedia.org/wiki/Actor_model)构建，其中 actor 是计算的基本单位。Actor 通过异步消息传递进行通信，提供：

- **位置透明性**：本地和远程 actor 使用相同 API
- **容错性**：Actor 可以独立失败，不会影响其他 actor
- **并发性**：Actor 一次处理一条消息，简化并发编程

### 主要特性

- **零外部依赖**：纯 Rust + Tokio 实现
- **内置服务发现**：SWIM/Gossip 协议管理集群
- **流式支持**：原生支持流式请求/响应
- **多语言**：Python 优先，Rust 核心，可扩展到其他语言

## API 风格

### Python API

Pulsing 提供多种 API 风格来适应不同用例：

#### 1. Actor System 风格（显式管理）

```python
import pulsing as pul

# 显式创建和管理 actor 系统
system = await pul.actor_system(addr="0.0.0.0:8000")

# 生成 actor
actor = await system.spawn(MyActor(), name="my_actor")

# 通信
response = await actor.ask({"message": "hello"})

# 关闭
await system.shutdown()
```

#### 2. Ray 风格全局 API（便捷）

```python
import pulsing as pul

# 初始化全局系统
await pul.init(addr="0.0.0.0:8000")

# 使用全局系统生成 actor
actor = await pul.spawn(MyActor(), name="my_actor")

# 通信
response = await actor.ask({"message": "hello"})

# 关闭
await pul.shutdown()
```

#### 3. Ray 兼容 API（迁移）

```python
from pulsing.compat import ray

# Ray 兼容 API，方便迁移
ray.init(address="0.0.0.0:8000")

@ray.remote
class MyActor:
    def process(self, data):
        return f"Processed: {data}"

actor = MyActor.remote()
result = ray.get(actor.process.remote("hello"))

ray.shutdown()
```

### Actor 模式

#### Remote 装饰器（推荐）

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, init=0):
        self.value = init

    # 同步方法 - 串行执行
    def incr(self):
        self.value += 1
        return self.value

    # 异步方法 - 并发执行
    async def fetch_and_add(self, url):
        data = await http_get(url)
        self.value += data
        return self.value

# 使用
counter = await Counter.spawn(name="counter")
result = await counter.incr()
```

#### 基础 Actor 类

```python
from pulsing.actor import Actor

class MyActor(Actor):
    async def receive(self, msg):
        if msg.get("action") == "greet":
            return f"Hello, {msg.get('name', 'World')}!"
        return "Unknown action"

# 使用
system = await pul.actor_system()
actor = await system.spawn(MyActor(), name="greeter")
response = await actor.ask({"action": "greet", "name": "Alice"})
```

### 消息传递

#### Ask vs Tell

- **`ask(msg)`**：请求/响应模式，等待并返回响应
- **`tell(msg)`**：发射后不管模式，发送消息不等待

```python
# Ask - 获取响应
response = await actor.ask({"action": "compute", "data": [1, 2, 3]})

# Tell - 无需响应
await actor.tell({"action": "log", "level": "info", "message": "Event occurred"})
```

### 流式响应

Pulsing 支持流式响应，用于大数据或持续生成：

```python
@pul.remote
class StreamingService:
    async def generate_tokens(self, prompt):
        for token in generate_tokens(prompt):
            yield token

# 使用
service = await StreamingService.spawn()
async for token in service.generate_tokens("Hello world"):
    print(token, end="")
```

### 监督与容错

Actor 可以配置重启策略以实现容错：

```python
@pul.remote(
    restart_policy="on_failure",  # "never", "on_failure", "always"
    max_restarts=3,
    min_backoff=0.1,
    max_backoff=30.0
)
class ResilientWorker:
    def process(self, data):
        # 如果抛出异常，Actor 会自动重启
        return risky_computation(data)
```

### 分布式队列

Pulsing 包含分布式队列系统，用于数据管道：

```python
# 写入
writer = await system.queue.write("my_topic", bucket_column="user_id")
await writer.put({"user_id": "u1", "data": "hello"})
await writer.flush()

# 读取
reader = await system.queue.read("my_topic")
records = await reader.get(limit=100)
```

## Rust API

### 核心 Trait

Rust API 通过 trait 定义契约，分为三层：

#### ActorSystemCoreExt（主路径，prelude 自动导入）

```rust
use pulsing_actor::prelude::*;

// 生成 actor
let actor = system.spawn_named("services/echo", EchoActor).await?;

// 通信
let response = actor.ask(Ping(42)).await?;
```

#### ActorSystemAdvancedExt（高级：可重启监督）

Factory 模式生成，支持监督重启（仅命名 actor）：

```rust
let options = SpawnOptions::new()
    .supervision(SupervisionSpec::on_failure().max_restarts(3));

// 仅命名 actor 支持 supervision
system.spawn_named_factory("services/worker", || Ok(Worker::new()), options).await?;
```

#### ActorSystemOpsExt（运维/诊断/生命周期）

系统信息、集群成员、停止/关闭等：

```rust
system.node_id();
system.addr();
system.members().await;
system.all_named_actors().await;
system.stop("name").await?;
system.shutdown().await?;
```

### Behavior（类型安全，Akka Typed 风格）

- **核心**：`Behavior<M>` + `TypedRef<M>` + `BehaviorAction (Same/Become/Stop)`
- **约定**：`TypedRef<M>` 要求 `M: Serialize + DeserializeOwned + Send + 'static`

## 错误处理

### Python

```python
try:
    response = await actor.ask({"action": "process", "data": data})
except RuntimeError as e:
    # Actor 端异常作为 RuntimeError 传输
    print(f"Actor error: {e}")
except ConnectionError as e:
    # 网络错误
    print(f"Connection error: {e}")
except asyncio.TimeoutError as e:
    # 超时错误
    print(f"Timeout: {e}")
```

### Rust

```rust
use anyhow::Result;

match actor.ask(Ping(42)).await {
    Ok(response) => println!("Got: {:?}", response),
    Err(e) => println!("Error: {:?}", e),
}
```

## 安全考虑

### 信任边界

- **Pickle 载荷**在 Python-Python 通信中可能导致 RCE
- 生产环境使用 TLS
- 将集群视为经过认证的信任边界

### 网络安全

```python
# 启用 TLS
system = await pul.actor_system(
    addr="0.0.0.0:8000",
    passphrase="your-secret-passphrase"
)
```

## 性能特性

- **低延迟**：HTTP/2 传输与二进制序列化
- **高吞吐量**：异步运行时与高效任务调度
- **内存高效**：基于 actor 的并发，无需线程
- **可扩展**：Gossip 基础集群发现，适用于大型部署

## 后续步骤

- **[Python API](python.md)**: Python 接口完整文档
- **[Rust API](rust.md)**: Rust 接口完整文档
- **[示例](../../examples/)**: 工作代码示例
- **[指南](../../guide/)**: 深入指南和教程
