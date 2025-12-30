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
    @staticmethod
    def single(msg_type: str, payload: bytes) -> Message:
        """创建单条消息。"""
        pass

    @staticmethod
    def stream(msg_type: str, payload: bytes) -> Message:
        """创建流式消息。"""
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

### @as_actor

自动将类转换为 Actor。

```python
@as_actor
class MyActor:
    def __init__(self, value: int):
        self.value = value

    def get(self) -> int:
        return self.value

    async def process(self, data: str) -> dict:
        return {"result": data.upper()}
```

装饰后，类提供：

- `local(system, **kwargs) -> ActorRef`: 本地创建 actor
- `remote(system, **kwargs) -> ActorRef`: 远程创建 actor（单节点时回退到本地）

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
