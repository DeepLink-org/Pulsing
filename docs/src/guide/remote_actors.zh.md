# 远程 Actor 指南

在集群中使用 Actor 的指南，支持位置透明。

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

远程 Actor 调用可能因网络问题而失败：

```python
try:
    remote_ref = await system.resolve("worker")
    response = await remote_ref.ask(msg)
except Exception as e:
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
