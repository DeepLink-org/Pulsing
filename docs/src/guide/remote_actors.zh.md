# 远程 Actor 指南

在集群中使用 Actor 的指南，支持位置透明。

## 集群设置

### 启动种子节点

```python
from pulsing.actor import SystemConfig, create_actor_system

# Node 1: 启动种子节点
config = SystemConfig.with_addr("0.0.0.0:8000")
system = await create_actor_system(config)

# 生成公共 actor
await system.spawn(WorkerActor(), "worker", public=True)
```

### 加入集群

```python
# Node 2: 加入集群
config = SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["192.168.1.1:8000"])
system = await create_actor_system(config)

# 等待集群同步
await asyncio.sleep(1.0)
```

## 查找远程 Actor

### 使用 system.find()

```python
# 按名称查找 actor（搜索整个集群）
remote_ref = await system.find("worker")

if remote_ref:
    response = await remote_ref.ask(Message.single("request", b"data"))
```

### 检查 Actor 是否存在

```python
# 检查 actor 是否存在于集群中
if await system.has_actor("worker"):
    ref = await system.find("worker")
    # 使用 ref...
```

## 公共 vs 私有 Actor

### 公共 Actor

公共 Actor 对集群中的所有节点可见：

```python
# 公共 actor - 可被其他节点找到
await system.spawn(WorkerActor(), "worker", public=True)
```

### 私有 Actor

私有 Actor 仅本地可访问：

```python
# 私有 actor - 仅本地
await system.spawn(WorkerActor(), "local-worker", public=False)
```

## 位置透明性

相同的 API 适用于本地和远程 Actor：

```python
# 本地 actor
local_ref = await system.spawn(MyActor(), "local")

# 远程 actor（通过集群找到）
remote_ref = await system.find("remote-worker")

# 两者使用相同的 API
response1 = await local_ref.ask(msg)
response2 = await remote_ref.ask(msg)
```

## 错误处理

远程 Actor 调用可能因网络问题而失败：

```python
try:
    remote_ref = await system.find("worker")
    if remote_ref:
        response = await remote_ref.ask(msg)
    else:
        print("Actor 未找到")
except Exception as e:
    print(f"远程调用失败: {e}")
```

## 最佳实践

1. **等待集群同步**：加入集群后添加短暂延迟
2. **处理缺失的 actor**：始终检查 `find()` 是否返回 None
3. **集群通信使用公共 actor**：需要远程访问的 actor 设置 `public=True`
4. **处理网络错误**：在 try-except 块中包装远程调用
5. **使用超时**：考虑为远程调用添加超时

## 示例：分布式计数器

```python
@remote
class DistributedCounter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

# Node 1
system1 = await create_actor_system(SystemConfig.with_addr("0.0.0.0:8000"))
counter1 = await DistributedCounter.local(system1, init_value=0)
await system1.spawn(counter1, "counter", public=True)

# Node 2
system2 = await create_actor_system(
    SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["127.0.0.1:8000"])
)
await asyncio.sleep(1.0)

# 从 Node 2 访问远程计数器
remote_counter = await system2.find("counter")
if remote_counter:
    value = await remote_counter.get()  # 0
    value = await remote_counter.increment(5)  # 5
```

## 下一步

- 了解 [Actor 系统](actor_system.zh.md) 基础知识
- 查看[节点发现](../design/node-discovery.md)了解集群详情
