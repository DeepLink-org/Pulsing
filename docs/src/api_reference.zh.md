# API 参考

Pulsing Actor 框架的完整 API 文档。

## 核心类

### Actor

所有 actor 的基类。

```python
class Actor:
    async def receive(self, msg: Message) -> Message:
        """处理传入消息。"""
        pass
```

### Message

Actor 通信的消息包装器。

```python
class Message:
    @property
    def msg_type(self) -> str:
        """获取消息类型。"""
        pass

    @property
    def payload(self) -> bytes:
        """获取原始负载字节。"""
        pass

    @property
    def is_stream(self) -> bool:
        """检查是否为流式消息。"""
        pass

    @staticmethod
    def single(msg_type: str, payload: bytes) -> Message:
        """创建带原始字节的单条消息。"""
        pass

    def to_json(self) -> Any:
        """将负载反序列化为 JSON。"""
        pass

    def to_object(self) -> Any:
        """将负载反序列化为 Python 对象（pickle）。"""
        pass

    def stream_reader(self) -> StreamReader:
        """获取流式消息的 StreamReader。"""
        pass
```

### StreamMessage

创建流式响应的工厂类。

```python
class StreamMessage:
    @staticmethod
    def create(
        msg_type: str = "",
        buffer_size: int = 32
    ) -> tuple[Message, StreamWriter]:
        """
        创建流式消息及其写入器。

        参数：
            msg_type: 流块的默认消息类型
            buffer_size: 有界通道缓冲区大小（背压控制）

        返回：
            (Message, StreamWriter) 元组
        """
        pass
```

### StreamWriter

流式响应的写入器。支持自动 Python 对象序列化。

```python
class StreamWriter:
    async def write(self, obj: Any) -> None:
        """
        将 Python 对象写入流。

        对象会自动使用 pickle 序列化，
        使 Python 到 Python 的流式传输完全透明。

        参数：
            obj: 任何可 pickle 的 Python 对象（dict、list、str 等）
        """
        pass

    async def close(self) -> None:
        """正常关闭流。"""
        pass

    async def error(self, message: str) -> None:
        """带错误关闭流。"""
        pass
```

### StreamReader

流式响应的读取器。自动反序列化 Python 对象。

```python
class StreamReader:
    async def __anext__(self) -> Any:
        """
        从流中获取下一个元素。

        直接返回 Python 对象（自动反序列化）。
        流结束时抛出 StopAsyncIteration。
        """
        pass

    def __aiter__(self) -> StreamReader:
        """返回自身作为异步迭代器。"""
        pass
```

### SystemConfig

Actor System 的配置。

```python
class SystemConfig:
    @staticmethod
    def standalone() -> SystemConfig:
        """创建单机（非集群）配置。"""
        pass

    @staticmethod
    def with_addr(addr: str) -> SystemConfig:
        """创建带地址的配置。"""
        pass

    def with_seeds(self, seeds: List[str]) -> SystemConfig:
        """添加用于集群发现的种子节点。"""
        pass
```

### ActorSystem

Actor 系统的主入口点。

```python
class ActorSystem:
    async def spawn(
        self,
        actor: Actor,
        name: str,
        public: bool = False
    ) -> ActorRef:
        """生成新的 actor。"""
        pass

    async def find(self, name: str) -> Optional[ActorRef]:
        """在集群中按名称查找 actor。"""
        pass

    async def has_actor(self, name: str) -> bool:
        """检查 actor 是否存在。"""
        pass

    async def shutdown(self) -> None:
        """关闭 actor 系统。"""
        pass
```

### ActorRef

Actor 的引用（本地或远程）。

```python
class ActorRef:
    async def ask(self, msg: Message) -> Message:
        """发送消息并等待响应。"""
        pass

    async def tell(self, msg: Message) -> None:
        """发送消息但不等待响应。"""
        pass

    async def ask_stream(self, msg: Message) -> AsyncIterator[Message]:
        """发送流式消息。"""
        pass
```

## 装饰器

### @remote

自动将类转换为 Actor。

```python
from pulsing.actor import init, shutdown, remote

@remote
class MyActor:
    def __init__(self, value: int):
        self.value = value

    def get(self) -> int:
        return self.value

    async def process(self, data: str) -> dict:
        return {"result": data.upper()}

async def main():
    await init()
    actor = await MyActor.spawn(value=10)
    print(await actor.get())  # 10
    await shutdown()
```

#### 监督（actor 级别重启）

`@remote` 支持通过可选参数配置 **actor 级别重启**：

- `restart_policy`：`"never"`（默认）、`"always"`、`"on-failure"`
- `max_restarts`：最大重启次数（默认 `3`）
- `min_backoff` / `max_backoff`：退避时间下限/上限（单位秒）

示例：

```python
from pulsing.actor import remote

@remote(restart_policy="on-failure", max_restarts=5, min_backoff=0.2, max_backoff=10.0)
class Worker:
    def work(self, x: int) -> int:
        return 100 // x
```

说明：

- 这**不是** supervision tree。
- 重启也**不等于** exactly-once；业务逻辑需要幂等与去重。

## 辅助函数

### ask_with_timeout

为 `ActorRef.ask()` 提供一个带超时的便捷封装：

```python
from pulsing.actor import ask_with_timeout

result = await ask_with_timeout(ref, {"op": "compute"}, timeout=10.0)
```


装饰后，类提供：

- `spawn(**kwargs) -> ActorRef`: 创建 actor（使用 `init()` 初始化的全局系统）

## 函数

### create_actor_system

创建新的 Actor System 实例。

```python
async def create_actor_system(config: SystemConfig) -> ActorSystem:
    """创建并启动 actor 系统。"""
    pass
```

## 示例

查看[快速开始指南](quickstart/index.zh.md)了解使用示例。
