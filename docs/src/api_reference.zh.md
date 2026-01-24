# API 参考

Pulsing Actor 框架的完整 API 文档。

## 契约与语义（由 `llms.binding.md` 派生）

本节是 Pulsing Python API 的**面向用户契约**，内容**派生自**仓库根目录的 **`llms.binding.md`**。

- **权威来源**：`llms.binding.md` 是对外契约源文件。
- **本文档**：API 参考 + 显式语义声明（并发、错误、信任边界）。
- **若两者不一致**：以 `llms.binding.md` 为准，并请提 Issue/PR 同步。

### `@pulsing.remote` 的并发语义

对 `@pulsing.remote` 类，方法调用会被翻译为 actor 消息并在 actor 内执行。

- **同步方法（`def method`）**
  - 在 actor 内**串行执行**（一次处理一个请求）。
  - 适合：快速计算、状态修改。
- **异步方法（`async def method`）**
  - 调用走**基于流（stream）的执行路径**，在 actor 侧以后台任务调度。
  - 当方法处于 `await` 等待期间，actor 仍可继续处理其他消息（即“非阻塞 actor”语义）。
  - 你既可以：
    - `await proxy.async_method(...)` 获取最终返回值；也可以
    - `async for chunk in proxy.async_method(...): ...` 消费中间 `yield`。
- **生成器（同步/异步）**
  - 返回 generator（sync/async）会被当作**流式响应**。

### 流式与取消（cancellation）

- 流式响应通过 Pulsing stream message 实现；取消传播属于 **best-effort**。
- 调用方取消本地 `await/async for` 时，远端是否立即停止取决于传输层取消传播。

### `ask` 与 `tell`

- **`ask(msg)`**：请求-响应，返回值或抛异常。
- **`tell(msg)`**：fire-and-forget，不等待返回。

### 错误模型（当前行为）

- actor 内抛出的异常通常会在调用方表现为 **`RuntimeError(str(e))`**。
- 若使用超时封装（如 `asyncio.wait_for`），超时会抛 **`asyncio.TimeoutError`**。

注意：错误类型信息与远端堆栈不保证完整保留。

### 信任边界与安全声明

- **Python ↔ Python 默认使用 Pickle**：
  - 为了易用性，Python↔Python 对象载荷默认使用 **pickle**。
  - **风险**：对不可信数据 unpickle 可能导致任意代码执行（RCE）。
  - **建议**：仅在**可信网络/可信集群边界**内使用。
- **传输层安全（TLS）**：
  - 生产环境建议启用 TLS/mTLS，并把集群内部通信视为经过认证的信任边界。

### 队列语义（Distributed Queue）

- **分桶**：
  - 写端使用 `bucket_column` + `num_buckets` 决定 record 落到哪个 bucket。
  - 读端必须与写端保持一致的 `num_buckets`（以及 backend）。
- **归属（owner）**：
  - bucket 的 owner 基于集群成员一致性哈希计算；请求可能被重定向到 owner 节点。
- **后端**：
  - 默认内存后端；是否持久化取决于 backend。

## 核心函数

### pul.actor_system

创建新的 Actor System 实例。

```python
import pulsing as pul

system = await pul.actor_system(
    addr: str | None = None,        # 绑定地址，None 为单机模式
    *,
    seeds: list[str] | None = None, # 集群种子节点
    passphrase: str | None = None,  # TLS 密码短语
) -> ActorSystem
```

**示例：**

```python
# 单机模式
system = await pul.actor_system()

# 集群模式
system = await pul.actor_system(addr="0.0.0.0:8000")

# 加入现有集群
system = await pul.actor_system(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])

# 关闭
await system.shutdown()
```

### pul.init / pul.shutdown

全局系统初始化（Ray 风格异步 API）。

```python
import pulsing as pul

# 初始化全局系统
await pul.init(addr=None, seeds=None, passphrase=None)

# 使用全局系统
actor = await pul.spawn(MyActor())
ref = await pul.resolve("actor_name")

# 关闭
await pul.shutdown()
```

## 核心类

### ActorSystem

Actor 系统的主入口点。

```python
class ActorSystem:
    async def spawn(
        self,
        actor: Actor,
        *,
        name: str | None = None,
        # public 参数已废弃：所有命名 actor 自动可被 resolve
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0
    ) -> ActorRef:
        """
        生成新的 actor。

        - 有 name: 命名 actor，可通过 resolve() 发现
        - 无 name: 匿名 actor，仅通过返回的 ActorRef 访问
        """
        pass

    async def refer(self, actorid: ActorId | str) -> ActorRef:
        """通过 ActorId 获取 ActorRef。"""
        pass

    async def resolve(self, name: str, *, node_id: int | None = None) -> ActorRef:
        """通过名称解析 actor。"""
        pass

    async def shutdown(self) -> None:
        """关闭 actor 系统。"""
        pass
```

### ActorRef

Actor 的底层引用。使用 `ask()` 和 `tell()` 进行通信。

```python
class ActorRef:
    @property
    def actor_id(self) -> ActorId:
        """获取 actor 的 ID。"""
        pass

    async def ask(self, msg: Any) -> Any:
        """发送消息并等待响应。"""
        pass

    async def tell(self, msg: Any) -> None:
        """发送消息但不等待响应（fire-and-forget）。"""
        pass
```

### ActorProxy

`@remote` 类的高级代理。可直接调用方法。

```python
class ActorProxy:
    @property
    def ref(self) -> ActorRef:
        """获取底层 ActorRef。"""
        pass

    # 直接调用方法：
    # result = await proxy.my_method(arg1, arg2)
```

## 装饰器

### @remote / @pul.remote

将类转换为分布式 Actor。

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    # 同步方法 - 顺序执行
    def incr(self) -> int:
        self.value += 1
        return self.value

    # 异步方法 - await 期间可并发执行
    async def fetch_and_add(self, url: str) -> int:
        data = await http_get(url)
        self.value += data
        return self.value

    # Generator - 自动流式传输
    async def stream(self):
        for i in range(10):
            yield {"count": i}

# 创建 actor
counter = await Counter.spawn(name="counter")

# 直接调用方法
result = await counter.incr()

# 流式传输
async for chunk in counter.stream():
    print(chunk)

# 解析已有 actor
proxy = await Counter.resolve("counter")
```

**监督参数：**

```python
@pul.remote(
    restart_policy="on_failure",  # "never" | "on_failure" | "always"
    max_restarts=3,
    min_backoff=0.1,
    max_backoff=30.0,
)
class ResilientWorker:
    def work(self, data): ...
```

## 基础 Actor

需要底层控制时，可使用基础 Actor 类。

```python
class MyActor:
    def __init__(self):
        self.value = 0

    def on_start(self, actor_id):
        """Actor 启动时调用。"""
        print(f"Started: {actor_id}")

    async def receive(self, msg):
        """处理传入消息。"""
        if msg.get("action") == "add":
            self.value += msg.get("n", 1)
            return {"value": self.value}
        return {"error": "unknown action"}

# 生成
system = await pul.actor_system()
actor = await system.spawn(MyActor(), name="my_actor")

# 通过 ask/tell 通信
response = await actor.ask({"action": "add", "n": 10})
```

## 队列 API

用于数据管道的分布式队列。

```python
# 写入
writer = await system.queue.write(
    topic="my_queue",
    bucket_column="user_id",
    num_buckets=4,
)
await writer.put({"user_id": "u1", "data": "hello"})
await writer.flush()

# 读取
reader = await system.queue.read("my_queue")
records = await reader.get(limit=100)
```

## Ray 兼容

Ray 的直接替换。

```python
from pulsing.compat import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    def incr(self):
        self.value += 1
        return self.value

counter = Counter.remote()
result = ray.get(counter.incr.remote())

ray.shutdown()
```

## Rust API

Rust API 通过三层 trait 组织（均在 `pulsing_actor::prelude::*` 中 re-export）：

### ActorSystemCoreExt（主路径）

核心 spawn 与 resolve 操作：

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

// Resolve - 简洁 API
system.actor_ref(&actor_id).await?;            // 按 ActorId 获取
system.resolve(name).await?;                   // 按名称解析

// Resolve - Builder 模式（高级配置）
system.resolving()
    .node(node_id)                             // 可选：指定目标节点
    .policy(RoundRobinPolicy::new())           // 可选：负载均衡策略
    .filter_alive(true)                        // 可选：只选存活节点
    .resolve(name).await?;                     // 解析单个

system.resolving().list(name).await?;          // 获取所有实例
system.resolving().lazy(name)?;                // 懒解析（~5s TTL 自动刷新）
```

### ActorSystemAdvancedExt（高级：监督/重启）

Factory 模式 spawn，支持 supervision 重启（仅命名 actor）：

```rust
let options = SpawnOptions::new()
    .supervision(SupervisionSpec::on_failure().max_restarts(3));

// 仅命名 actor 支持 supervision（匿名 actor 无法重新解析）
system.spawn_named_factory(name, || Ok(Service::new()), options).await?;
```

### ActorSystemOpsExt（运维/诊断）

系统信息、集群成员、生命周期控制：

```rust
system.node_id();
system.addr();
system.members().await;
system.all_named_actors().await;
system.stop(name).await?;
system.shutdown().await?;
```

## 示例

查看[快速开始指南](quickstart/index.zh.md)了解使用示例。
