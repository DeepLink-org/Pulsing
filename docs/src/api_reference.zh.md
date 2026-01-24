# API 参考

Pulsing Actor 框架的完整 API 文档。

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
        public: bool = False,
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0
    ) -> ActorRef:
        """生成新的 actor。"""
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

## 示例

查看[快速开始指南](quickstart/index.zh.md)了解使用示例。
