# 远程 Actor 指南

在集群中运行与发现 Actor 的指南：集群搭建、命名 Actor、resolve。

## 前后对比：单节点 vs 集群

**同一套 Actor 代码。** 只有初始化以及获取引用的方式不同。

| | 单节点（独立） | 集群（两节点） |
|---|----------------|----------------|
| **初始化** | `await pul.init()` | 节点 1：`await pul.init(addr="0.0.0.0:8000")`<br/>节点 2：`await pul.init(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])` |
| **获取 Actor** | `await Counter.spawn(value=0)` | 节点 1：`await Counter.spawn(value=0, name="counter")`<br/>节点 2：`await Counter.resolve("counter")` |
| **调用** | `await counter.inc()` | 相同：`await counter.inc()` — 位置透明 |

一旦拿到 proxy（来自 `spawn` 或 `resolve`），API 完全一致，业务逻辑无需区分“远程”和“本地”。

---

## 集群设置

### 启动种子节点

```python
import pulsing as pul

# Node 1: 启动种子节点
system = await pul.actor_system(addr="0.0.0.0:8000")

# 生成命名 actor（可通过 resolve 发现）
await system.spawn(WorkerActor(), name="worker")
```

### 加入集群

```python
# Node 2: 加入集群
system = await pul.actor_system(
    addr="0.0.0.0:8001",
    seeds=["192.168.1.1:8000"]
)

# 等待集群同步
await asyncio.sleep(1.0)
```

## 查找远程 Actor

### 使用 system.resolve()

```python
# 按名称查找 actor（搜索整个集群）
remote_ref = await system.resolve("worker")
response = await remote_ref.ask({"action": "process", "data": "hello"})

# 将 ActorRef 转换为代理
any_proxy = remote_ref.as_any()           # 未知类型时使用
typed_proxy = remote_ref.as_type(Worker)  # 已知类型时使用
```

### 使用 @remote 类的 resolve()

```python
@pul.remote
class Worker:
    def process(self, data): return f"processed: {data}"

# 带类型信息解析 - 返回带方法的 ActorProxy
worker = await Worker.resolve("worker")
result = await worker.process("hello")  # 直接调用方法
```

!!! note
    新代码优先使用 `Class.resolve(name)`（typed proxy）。仅在只有运行时名称时使用 `system.resolve(name)`，随后对返回的 `ActorRef` 调用 `.as_type()` / `.as_any()`。

## 命名 vs 匿名 Actor

### 命名 Actor（可发现）

命名 Actor 在集群中可被任意节点通过 `resolve()` 发现：

```python
# 命名 actor - 可通过 resolve() 从任意节点发现
await system.spawn(WorkerActor(), name="worker")

# 其他节点可以通过名称找到
ref = await other_system.resolve("worker")
```

### 匿名 Actor（仅本地引用）

匿名 Actor 只能通过 spawn 返回的 ActorRef 访问：

```python
# 匿名 actor - 仅通过 ActorRef 访问
local_ref = await system.spawn(WorkerActor())

# 无法通过 resolve() 找到，只能使用返回的 ActorRef
await local_ref.ask(msg)
```

## 位置透明性

命名 Actor 支持位置透明 —— 相同的 API 适用于本地和远程：

```python
# 本地命名 actor
local_ref = await system.spawn(MyActor(), name="local-worker")

# 远程命名 actor（通过集群 resolve）
remote_ref = await system.resolve("remote-worker")

# 两者使用完全相同的 API
response1 = await local_ref.ask(msg)
response2 = await remote_ref.ask(msg)
```

## 错误处理

Pulsing 为本地和远程 Actor 提供了统一的错误类型，确保在集群中一致的错误处理。

### 错误类型

- **PulsingRuntimeError**: 框架错误（网络、集群、Actor 系统等）
- **PulsingActorError**: Actor 执行错误
  - **PulsingBusinessError**: 业务逻辑错误（用户输入验证等）
  - **PulsingSystemError**: 系统错误（可能触发 Actor 重启）
  - **PulsingTimeoutError**: 超时错误（可重试）

### 示例

```python
from pulsing.exceptions import (
    PulsingBusinessError,
    PulsingSystemError,
    PulsingRuntimeError,
)

try:
    remote_ref = await system.resolve("worker")
    response = await remote_ref.ask(msg)
except PulsingBusinessError as e:
    # 处理业务错误（用户输入问题）
    print(f"验证失败: {e.message}")
except PulsingSystemError as e:
    # 处理系统错误（可能触发重启）
    print(f"系统错误: {e.error}, 可恢复: {e.recoverable}")
except PulsingRuntimeError as e:
    # 处理框架错误（网络、集群等）
    print(f"框架错误: {e}")
```

### 网络故障

网络相关错误会作为 `PulsingRuntimeError` 抛出：

```python
try:
    remote_ref = await system.resolve("worker")
    response = await remote_ref.ask(msg)
except PulsingRuntimeError as e:
    # 网络故障、集群问题或 Actor 未找到
    if "Connection" in str(e) or "timeout" in str(e).lower():
        # 使用退避策略重试
        pass
    elif "not found" in str(e).lower():
        # Actor 不存在
        pass
```

### 超时

为远程调用使用超时，避免无限等待：

```python
from pulsing.core import ask_with_timeout

try:
    response = await ask_with_timeout(remote_ref, msg, timeout=10.0)
except asyncio.TimeoutError:
    print("请求超时")
except PulsingRuntimeError as e:
    print(f"远程调用失败: {e}")
```

## 最佳实践

1. **等待集群同步**：加入集群后添加短暂延迟
2. **优雅处理错误**：在 try-except 块中包装远程调用
3. **使用命名 actor**：需要远程访问的 actor 必须有 `name`
4. **使用 @remote 与 resolve()**：获取有类型的代理以获得更好的 API 体验
5. **使用超时**：考虑为远程调用添加超时

## 示例：分布式计数器

```python
import pulsing as pul

@pul.remote
class DistributedCounter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

# Node 1: 创建命名计数器（可被远程发现）
system1 = await pul.actor_system(addr="0.0.0.0:8000")
counter = await DistributedCounter.spawn(name="counter", init_value=0)

# Node 2: 访问远程计数器
system2 = await pul.actor_system(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])
await asyncio.sleep(1.0)

# 解析并使用远程计数器
remote_counter = await DistributedCounter.resolve("counter")
value = await remote_counter.get()  # 0
value = await remote_counter.increment(5)  # 5
```

## 下一步

- 了解 [Actor 系统](actor_system.zh.md) 基础知识
- 查看[节点发现](../design/node-discovery.zh.md)了解集群详情
