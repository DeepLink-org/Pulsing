# Actor 系统指南

本指南介绍 Pulsing Actor 系统的核心概念和用法。

## 什么是 Actor？

Actor 是一个计算单元，具有以下特性：
- 封装状态和行为
- 仅通过消息通信
- 异步处理消息
- 具有唯一标识符

## 创建 Actor

### 使用 Actor 基类

```python
from pulsing.actor import Actor, Message, SystemConfig, create_actor_system

class MyActor(Actor):
    def __init__(self):
        self.state = {}

    async def receive(self, msg: Message) -> Message:
        # 处理消息
        return Message.single("response", b"data")

async def main():
    system = await create_actor_system(SystemConfig.standalone())
    actor_ref = await system.spawn(MyActor(), "my-actor")
    await system.shutdown()
```

### 使用 @as_actor 装饰器

`@as_actor` 装饰器可以自动将类转换为 Actor：

```python
from pulsing.actor import as_actor, SystemConfig, create_actor_system

@as_actor
class Counter:
    def __init__(self, value: int = 0):
        self.value = value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

async def main():
    system = await create_actor_system(SystemConfig.standalone())
    counter = await Counter.local(system, value=10)
    print(await counter.get())  # 10
    await system.shutdown()
```

## 消息传递

### Ask 模式（请求-响应）

```python
response = await actor_ref.ask(Message.single("ping", b"data"))
```

### Tell 模式（发送即忘）

```python
await actor_ref.tell(Message.single("notify", b"data"))
```

### 流式消息

```python
msg = Message.stream("process", b"data")
async for chunk in actor_ref.ask_stream(msg):
    print(f"Received: {chunk}")
```

## Actor 生命周期

Actor 具有简单的生命周期：
1. **创建**：Actor 在系统中生成
2. **活跃**：Actor 处理消息
3. **停止**：Actor 从系统中移除

```python
# 创建 actor
actor_ref = await system.spawn(MyActor(), "my-actor")

# Actor 现在处于活跃状态，可以接收消息
response = await actor_ref.ask(msg)

# 停止 actor
await system.stop("my-actor")
```

## 状态管理

Actor 封装状态。每个 actor 实例都有自己独立的状态：

```python
@as_actor
class StatefulActor:
    def __init__(self):
        self.counter = 0
        self.data = {}

    def increment(self) -> int:
        self.counter += 1
        return self.counter

    def set_data(self, key: str, value: str) -> None:
        self.data[key] = value

    def get_data(self, key: str):
        return self.data.get(key)
```

## 错误处理

Actor 消息处理中的错误会传播给调用者：

```python
try:
    response = await actor_ref.ask(msg)
except Exception as e:
    print(f"Actor 错误: {e}")
```

## 最佳实践

1. **保持 Actor 专注**：每个 Actor 应该只有一个职责
2. **使用不可变数据**：尽可能使用不可变数据结构
3. **优雅处理错误**：始终处理消息处理中的潜在错误
4. **使用异步方法**：对于 I/O 操作，使用异步方法
5. **清理关闭**：完成后始终调用 `system.shutdown()`

## 下一步

- 了解[远程 Actor](remote_actors.zh.md)进行集群通信
- 查看[设计文档](../design/actor-system.md)了解实现细节
