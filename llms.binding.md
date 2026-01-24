# Pulsing API Reference for LLMs

## Overview

`Pulsing` is a distributed actor framework that provides a communication backbone for building distributed systems, with specialized support for AI applications.

## Python 接口

### Actor System风格接口

```Python
import pulsing as pul

system = await pul.actor_system(
    addr: str | None = None,
    *,
    seeds: list[str] | None = None,
    passphrase: str | None = None
) -> ActorSystem

await system.shutdown()

class MyActor:
    async def receive(self, msg: Any) -> Any:
        ...

actorref = await system.spawn(
    actor: Actor, # MyActor()
    *,
    name: str | None = None,
    public: bool = False,
    restart_policy: str = "never",
    max_restarts: int = 3,
    min_backoff: float = 0.1,
    max_backoff: float = 30.0
) -> ActorRef

actorref = await system.refer(actorid: ActorId | str) -> ActorRef

actorref = await system.resolve(
    name: str,
    *,
    node_id: int | None = None
) -> ActorRef

response = await actorref.ask(request: Any) -> Any

await actorref.tell(msg: Any) -> None


@pul.remote
class Counter:
    # 同步处理函数
    def incr(self):
        ...

    # 异步处理函数
    async def desc(self):
        ...

# 使用
counter = await Counter.spawn(name="counter")
result = await counter.incr()  # 返回 ActorProxy，直接调用方法

# 队列接口
writer = await system.queue.write(
    topic: str,
    *,
    bucket_column: str = "id",
    num_buckets: int = 4,
    batch_size: int = 100,
    storage_path: str | None = None,
    backend: str = "memory",
) -> QueueWriter

await writer.put(record: dict | list[dict]) -> None
await writer.flush() -> None

reader = await system.queue.read(
    topic: str,
    *,
    bucket_id: int | None = None,
    bucket_ids: list[int] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    num_buckets: int = 4,
) -> QueueReader

records = await reader.get(limit: int = 100, wait: bool = False) -> list[dict]

# 队列使用示例
writer = await system.queue.write("my_queue", bucket_column="user_id")
await writer.put({"user_id": "u1", "data": "hello"})

reader = await system.queue.read("my_queue")
records = await reader.get(limit=10)
```

### Ray风格异步接口

```python
import pulsing as pul

# 初始化全局系统
await pul.init(
    addr: str | None = None,
    *,
    seeds: list[str] | None = None,
    passphrase: str | None = None
) -> ActorSystem

await pul.shutdown()

# 生成 Actor（使用全局系统）
actorref = await pul.spawn(
    actor: Actor,
    *,
    name: str | None = None,
    public: bool = False,
    restart_policy: str = "never",
    max_restarts: int = 3,
    min_backoff: float = 0.1,
    max_backoff: float = 30.0
) -> ActorRef

# 通过 ActorId 获取引用（使用全局系统）
actorref = await pul.refer(actorid: ActorId | str) -> ActorRef

# 通过名称解析 Actor（使用全局系统）
actorref = await pul.resolve(
    name: str,
    *,
    node_id: int | None = None
) -> ActorRef

# 发送消息并等待响应
response = await actorref.ask(request: Any) -> Any

# 发送消息（不等待响应）
await actorref.tell(msg: Any) -> None

# 将 ActorRef 绑定到类型，生成 ActorProxy
proxy = Counter.resolve(name)

@pul.remote
class Counter:
    def __init__(self, init=0): self.value = init

    # 同步处理函数
    def incr(self):
        ...

    # 异步处理函数
    async def desc(self):
        ...

# 使用方式1：通过 spawn 创建
counter = await Counter.spawn(name="counter")
result = await counter.incr()  # 返回 ActorProxy，直接调用方法

# 使用方式2：通过 resolve 解析已有 actor
proxy = await Counter.resolve("counter")
result = await proxy.incr()

```

### Ray风格兼容接口

```python
from pulsing.compat import ray

# 初始化（同步接口，内部使用异步）
ray.init(
    address: str | None = None,
    *,
    ignore_reinit_error: bool = False,
    **kwargs
) -> None

# 关闭系统
ray.shutdown() -> None

# 检查是否已初始化
ray.is_initialized() -> bool

# 装饰器：将类转换为 Actor
@ray.remote
class MyActor:
    def __init__(self, ...): ...
    def method(self, ...): ...

# 创建 Actor（同步接口）
actor_handle = MyActor.remote(...) -> _ActorHandle

# 调用方法（返回 ObjectRef）
result_ref = actor_handle.method.remote(...) -> ObjectRef

# 获取结果（同步接口，支持单个或列表）
result = ray.get(
    refs: ObjectRef | list[ObjectRef],
    *,
    timeout: float | None = None
) -> Any | list[Any]

# 将值包装为 ObjectRef（用于 API 兼容）
ref = ray.put(value: Any) -> ObjectRef

# 等待多个 ObjectRef 完成
ready, remaining = ray.wait(
    refs: list[ObjectRef],
    *,
    num_returns: int = 1,
    timeout: float | None = None
) -> tuple[list[ObjectRef], list[ObjectRef]]
```

### Actor 行为

#### 基础 Actor（使用 `receive` 方法）

```python
from pulsing.actor import Actor

class EchoActor(Actor):
    """receive 方法 - 同步或异步均可，框架自动检测"""

    # 方式1：同步方法
    def receive(self, msg):
        return msg

    # 方式2：异步方法（需要 await 时使用）
    async def receive(self, msg):
        result = await some_async_operation()
        return result

class FireAndForget(Actor):
    """无返回值（适合 tell 调用）"""
    def receive(self, msg):
        print(f"Received: {msg}")
        # 无返回值
```

**注意：** `receive` 方法可以是 `def` 或 `async def`，Pulsing 会自动检测并正确处理。
只有当方法内部需要 `await` 其他协程时，才需要使用 `async def`。

#### @pul.remote 装饰器（推荐）

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, init=0):
        self.value = init

    # 同步方法 - 阻塞处理，请求按顺序执行
    # 适合：快速计算、状态修改
    def incr(self):
        self.value += 1
        return self.value

    # 异步方法 - 非阻塞，可并发处理多个请求
    # 适合：IO 密集型操作（网络请求、数据库查询）
    async def fetch_and_add(self, url):
        data = await http_get(url)  # 等待期间可处理其他请求
        self.value += data
        return self.value

    # 无返回值方法 - 适合 tell() 调用
    def reset(self):
        self.value = 0

# 同步 vs 异步方法的并发行为：
# - def method():     阻塞 Actor，请求排队顺序执行
# - async def method(): 非阻塞，await 期间可处理其他请求（并发）

# 使用
counter = await Counter.spawn(name="counter")
result = await counter.incr()      # ask 模式，等待返回
await counter.reset()              # 无返回值，但仍等待完成
```

#### 消息传递模式

```python
# ask - 发送消息并等待响应
response = await actorref.ask({"action": "get"})

# tell - 发送消息，不等待响应（fire-and-forget）
await actorref.tell({"action": "log", "data": "hello"})
```

#### Actor 生命周期

```python
from pulsing.actor import Actor, ActorId

class MyActor(Actor):
    def on_start(self, actor_id: ActorId):
        """Actor 启动时调用"""
        print(f"Started: {actor_id}")

    def on_stop(self):
        """Actor 停止时调用"""
        print("Stopping...")

    def metadata(self) -> dict[str, str]:
        """返回 Actor 元数据（用于诊断）"""
        return {"type": "worker", "version": "1.0"}

    async def receive(self, msg):
        return msg
```

#### 监督与重启策略

```python
@pul.remote(
    restart_policy="on_failure",  # "never" | "on_failure" | "always"
    max_restarts=3,               # 最大重启次数
    min_backoff=0.1,              # 最小退避时间（秒）
    max_backoff=30.0,             # 最大退避时间（秒）
)
class ResilientWorker:
    def process(self, data):
        # 如果抛出异常，Actor 会自动重启
        return heavy_computation(data)
```

#### 流式响应

```python
@pul.remote
class StreamingService:
    # 直接返回 generator，Pulsing 自动处理为流式响应
    async def generate_stream(self, n):
        for i in range(n):
            yield f"chunk_{i}"

    # 同步 generator 也支持
    def sync_stream(self, n):
        for i in range(n):
            yield f"item_{i}"

# 使用
service = await StreamingService.spawn()

# 客户端消费流
async for chunk in service.generate_stream(10):
    print(chunk)  # chunk_0, chunk_1, ...
```

**注意：** 对于 `@pul.remote` 类，直接返回 generator（同步或异步）即可，Pulsing 会自动检测并按流式响应处理。

## Rust 接口

Rust API 通过 trait 定义契约，分为三层：

### 快速入门

```rust
use pulsing_actor::prelude::*;

#[derive(Serialize, Deserialize)]
struct Ping(i32);

#[derive(Serialize, Deserialize)]
struct Pong(i32);

struct Echo;

#[async_trait]
impl Actor for Echo {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let Ping(x) = msg.unpack()?;
        Message::pack(&Pong(x))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    // 命名 actor（可通过 resolve 发现，使用 namespace/name 格式）
    let actor = system.spawn_named("services/echo", Echo).await?;
    let Pong(x): Pong = actor.ask(Ping(1)).await?;

    // 匿名 actor（仅通过 ActorRef 访问）
    let worker = system.spawn(Worker::new()).await?;

    system.shutdown().await?;
    Ok(())
}
```

### Trait 分层

#### ActorSystemCoreExt（主路径，prelude 自动导入）

核心 spawn 与 resolve 能力：

```rust
// Spawn - 简洁 API
system.spawn(actor).await?;                    // 匿名 actor（不可 resolve）
system.spawn_named(name, actor).await?;        // 命名 actor（可 resolve）

// Spawn - Builder 模式（高级配置）
system.spawning()
    .name("services/counter")                  // 可选：有 name = 可 resolve
    .supervision(SupervisionSpec::on_failure().max_restarts(3))
    .mailbox_capacity(256)
    .spawn(actor).await?;

// Resolve - 简单方式
system.actor_ref(&actor_id).await?;            // 按 ActorId 获取
system.resolve(name).await?;                   // 按名称解析

// Resolve - Builder 模式（高级配置）
system.resolving()
    .node(node_id)                             // 可选：指定目标节点
    .policy(RoundRobinPolicy::new())           // 可选：负载均衡策略
    .filter_alive(true)                        // 可选：只选存活节点
    .resolve(name).await?;                     // 解析单个

system.resolving()
    .list(name).await?;                        // 获取所有实例

system.resolving()
    .lazy(name)?;                              // 懒解析
```

#### ActorSystemAdvancedExt（高级：可重启 supervision）

Factory 模式 spawn，支持 supervision 重启（仅命名 actor）：

```rust
// 命名 actor + factory（可重启 + 可 resolve）
// 注意：匿名 actor 不支持 supervision，因为无法重新解析
system.spawn_named_factory(name, || Ok(Service::new()), options).await?;
```

#### ActorSystemOpsExt（运维/诊断/生命周期）

系统信息、集群成员、停止/关闭等：

```rust
system.node_id();
system.addr();
system.members().await;
system.all_named_actors().await;
system.stop(name).await?;
system.shutdown().await?;
```

### 关键约定

- **消息编码**：`Message::pack(&T)` 使用 bincode + `type_name::<T>()`；跨版本协议建议 `Message::single("TypeV1", bytes)`。
- **命名与解析**：
  - `spawn_named(name, actor)`：创建可发现 actor，name 即为解析路径
  - `resolve(name)`：一次性解析（迁移后可能 stale）
  - `resolve_lazy(name)`：懒解析 + 自动刷新（~5s TTL）
- **流式**：返回 `Message::Stream`，取消语义 best-effort。
- **监督**：只有 `spawn_named_factory` 支持失败重启，匿名 actor 不支持 supervision。

### Behavior（类型安全，Akka Typed 风格）

- **核心**：`Behavior<M>` + `TypedRef<M>` + `BehaviorAction (Same/Become/Stop)`
- **约定**：`TypedRef<M>` 要求 `M: Serialize + DeserializeOwned + Send + 'static`

除了定义时候使用函数语法以外，其他与 Actor 完全相同：

```rust
fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n, _ctx| {
        *count += n;
        BehaviorAction::Same
    })
}

// Behavior 实现 IntoActor trait，可以直接传给 spawn/spawn_named
// 无需手动包装，系统会自动转换
let counter = system.spawn(counter(0)).await?;
let counter = system.spawn_named("actors/counter", counter(0)).await?;
```
