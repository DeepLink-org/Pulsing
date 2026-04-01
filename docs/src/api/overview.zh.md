# API 概述

Pulsing 是分布式 AI 系统的通信骨干——一个具备流式支持、零依赖和内置发现的分布式 Actor 运行时。

## 核心概念

Pulsing 基于 [Actor 模型](https://en.wikipedia.org/wiki/Actor_model)构建，actor 是计算的基本单位，通过异步消息传递通信：

- **位置透明性**：本地和远程 actor 使用相同 API
- **容错性**：Actor 可以独立失败，不会影响其他 actor
- **并发性**：Actor 一次处理一条消息，简化并发编程

### 主要特性

- **零外部依赖**：纯 Rust + Tokio 实现
- **内置服务发现**：SWIM/Gossip 协议管理集群
- **流式支持**：原生支持流式请求/响应
- **兼容 subprocess 的执行能力**：`pulsing.subprocess` 保持标准库语义，并可按需通过 Pulsing 执行命令
- **多语言**：Python 优先，Rust 核心，可扩展到其他语言

## Quick Start

```python
import pulsing as pul

await pul.init()

@pul.remote
class Counter:
    def __init__(self): self.value = 0
    def incr(self): self.value += 1; return self.value

counter = await Counter.spawn(name="counter")
print(await counter.incr())  # 1

counter2 = await Counter.resolve("counter")
print(await counter2.incr())  # 2

await pul.shutdown()
```

## Python API

使用任何 API 之前，必须先调用 `await pul.init()`。

### 生命周期

```python
import pulsing as pul

await pul.init(
    addr=None,          # 绑定地址，单机模式为 None
    seeds=None,         # 集群种子节点
    passphrase=None,    # TLS 密码短语
)

await pul.shutdown()
```

### 定义 Actor

使用 `@pul.remote` 将任意类变为分布式 actor：

```python
@pul.remote
class Counter:
    def __init__(self, init=0):
        self.value = init

    def incr(self):                       # 同步方法 — 串行执行
        self.value += 1
        return self.value

    async def fetch_and_add(self, url):   # 异步方法 — await 期间可并发
        data = await http_get(url)
        self.value += data
        return self.value
```

### 创建与调用

`Class.spawn()` 创建 actor 并返回类型化代理：

```python
counter = await Counter.spawn(name="counter", init=10)
result = await counter.incr()             # 直接方法调用
```

### 解析已有 Actor

```python
# 类型化代理 — 已知 actor 类型时
proxy = await Counter.resolve("counter")
result = await proxy.incr()

# 类型化代理 — 手动绑定
proxy = await pul.resolve("counter", cls=Counter, timeout=30)

# 无类型代理 — 远端类型未知时
proxy = await pul.resolve("service_name")
result = await proxy.any_method(args)
```

### 流式响应

远程方法返回生成器即可进行流式传输：

```python
@pul.remote
class StreamingService:
    async def generate_tokens(self, prompt):
        for token in generate_tokens(prompt):
            yield token

service = await StreamingService.spawn()
async for token in service.generate_tokens("Hello world"):
    print(token, end="")
```

### 监督与容错

```python
@pul.remote(
    restart_policy="on_failure",  # "never", "on_failure", "always"
    max_restarts=3,
    min_backoff=0.1,
    max_backoff=30.0,
)
class ResilientWorker:
    def process(self, data):
        return risky_computation(data)
```

### 队列 (Queue)

分布式队列，支持 bucket 分区：

```python
writer = await pul.queue.write("my_queue", bucket_column="user_id")
await writer.put({"user_id": "u1", "data": "hello"})
await writer.flush()

reader = await pul.queue.read("my_queue")
records = await reader.get(limit=100)
```

### 主题 (Topic)

轻量级发布/订阅，用于实时消息分发：

```python
writer = await pul.topic.write("events")
await writer.publish({"type": "user_login", "user": "alice"})

reader = await pul.topic.read("events")

@reader.on_message
async def handle(msg):
    print(f"Received: {msg}")

await reader.start()
```

### Subprocess

可以使用与标准库兼容的同步 API 执行命令：

```python
import pulsing.subprocess as subprocess

result = subprocess.run(["echo", "hello"], capture_output=True, text=True)
remote = subprocess.run(
    ["hostname"],
    capture_output=True,
    text=True,
    resources={"num_cpus": 1},
)
```

不传 `resources` 时，行为等同于 Python 原生 `subprocess`。
传入 `resources` 且设置 `USE_POLSING_SUBPROCESS=1` 时，命令会通过
Pulsing 后端执行。详见 [子进程示例](../examples/subprocess.zh.md)。

### Under the Hood

#### ActorSystem（显式管理）

```python
import pulsing as pul

system = await pul.actor_system(addr="0.0.0.0:8000")

class MyActor:
    async def receive(self, msg):
        return f"echo: {msg}"

actor = await system.spawn(MyActor(), name="my_actor")
response = await actor.ask({"message": "hello"})
await actor.tell({"event": "fire_and_forget"})

await system.shutdown()
```

---

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
let options = SpawnOptions::default()
    .supervision(SupervisionSpec::on_failure().max_restarts(3));

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

---

## 错误处理

### Python

```python
from pulsing.exceptions import (
    PulsingBusinessError,
    PulsingSystemError,
    PulsingRuntimeError,
)

try:
    result = await service.process(data)
except PulsingBusinessError as e:
    print(f"业务错误 [{e.code}]: {e.message}")
except PulsingSystemError as e:
    print(f"系统错误: {e.error}, 可恢复: {e.recoverable}")
except PulsingRuntimeError as e:
    print(f"框架错误: {e}")
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

- **Pickle 载荷**在 Python-Python 通信中可能导致 RCE
- 生产环境使用 TLS
- 将集群视为经过认证的信任边界

```python
await pul.init(addr="0.0.0.0:8000", passphrase="your-secret-passphrase")
```

## 性能特性

- **低延迟**：HTTP/2 传输与二进制序列化
- **高吞吐量**：异步运行时与高效任务调度
- **内存高效**：基于 actor 的并发，无需线程
- **可扩展**：Gossip 基础集群发现，适用于大型部署

## 后续步骤

- **[Python API](python.md)**：Python 接口完整文档
- **[Rust API](rust.md)**：Rust 接口完整文档
- **[示例](../../examples/)**：工作代码示例
- **[指南](../../guide/)**：深入指南和教程
