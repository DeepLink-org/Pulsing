# Pulsing API Reference for LLMs

## Overview

`Pulsing`是一款分布式系统通信框架，可以作为任意分布式系统的通信骨架，以方便快速搭建分布式系统和应用。

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

# 装饰器：将类转换为 Actor
@ray.remote
class MyActor:
    def __init__(self, ...): ...
    def method(self, ...): ...

# 创建 Actor（同步接口）
actor_handle = MyActor.remote(...) -> _ActorHandle

# 调用方法（返回 ObjectRef）
result_ref = actor_handle.method.remote(...) -> ObjectRef

# 获取结果（同步接口）
result = ray.get(result_ref, timeout: float | None = None) -> Any

# 关闭系统
ray.shutdown() -> None
```
