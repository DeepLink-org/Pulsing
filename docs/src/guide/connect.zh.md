# 集群外连接 (Out-Cluster Connect)

**不加入集群**，直接访问 Pulsing 集群中的 actor。

`pulsing.connect` 提供一个轻量连接器，连接到集群的任意节点（作为 gateway），透明地路由请求到目标 actor。连接器不参与 gossip、不注册为节点、不运行故障检测——保持极致轻量。

## 适用场景

| 场景 | 为什么用 Connect？ |
|------|-------------------|
| **Notebook / CLI** | 只需要调用 actor，不需要托管 actor |
| **Web 后端** | 短生命周期请求不应触发集群成员变动 |
| **跨网络** | 调用方可能在不同的网络区域 |
| **安全隔离** | 外部调用方不应看到集群内部拓扑 |

!!! tip
    如果你的进程也需要**托管** actor（被发现、响应消息），请使用常规的 `pul.init()` + 集群加入。`Connect` 仅适用于**调用方**。

---

## 快速开始

```python
from pulsing.connect import Connect

# 连接到集群任意节点（作为 gateway）
conn = await Connect.to("10.0.1.1:8080")

# 解析命名 actor
counter = await conn.resolve("services/counter")

# 调用方法——与集群内 ActorProxy 语法完全一致
value = await counter.increment(5)
print(value)  # 5

# 完成后关闭连接
await conn.close()
```

就这么简单。不需要 `init()`，不需要集群配置，不需要开放端口。

---

## 通信范式

### 同步方法

```python
conn = await Connect.to("10.0.1.1:8080")
calc = await conn.resolve("services/calculator")

result = await calc.multiply(6, 7)   # 42
await calc.add(10)                    # 有状态调用
value = await calc.get()              # 10
```

### 异步方法

远程 actor 的异步方法透明工作：

```python
svc = await conn.resolve("services/ai")

# 异步方法——正常 await 即可
result = await svc.slow_process(data)
greeting = await svc.greet("world")  # "hello world"
```

### 流式传输（异步生成器）

适用于增量返回结果的 actor（如 LLM token 生成）：

```python
llm = await conn.resolve("services/llm")

# 逐 token 流式接收
async for token in llm.generate(prompt="讲个故事"):
    print(token, end="", flush=True)
```

!!! note
    流式使用 `async for proxy.method(args)` 直接迭代——`async for` 前面**不需要 `await`**。方法调用直接返回异步可迭代对象。

### 远程 Spawn

通过连接器在集群节点上创建新 actor——actor 运行在集群节点上，而非本地：

```python
from pulsing.connect import Connect
from myapp.actors import Calculator, Worker

conn = await Connect.to("10.0.1.1:8080")

# 带构造参数 spawn
calc = await conn.spawn(Calculator, init_value=100, name="services/calc")
result = await calc.multiply(6, 7)  # 42

# spawn 异步 actor
worker = await conn.spawn(Worker, name="services/worker")
status = await worker.process("task_data")
```

!!! note
    `@remote` 修饰的类必须在目标集群节点上已注册。即集群进程必须已导入该类定义。

spawn 后的 actor 也可以通过 `resolve()` 访问：

```python
# 其他连接器（或同一个）可以 resolve 已 spawn 的 actor
calc = await conn.resolve("services/calc")
await calc.add(50)
```

### 并发调用

支持并发发起多个调用：

```python
import asyncio

svc = await conn.resolve("services/worker")
results = await asyncio.gather(
    svc.process("task_a"),
    svc.process("task_b"),
    svc.process("task_c"),
)
```

---

## 多 Gateway（高可用）

传入地址列表实现自动故障切换：

```python
conn = await Connect.to([
    "10.0.1.1:8080",
    "10.0.1.2:8080",
    "10.0.1.3:8080",
])

# 当前 gateway 不可用时，连接器自动切换到下一个
```

随时从集群刷新 gateway 列表：

```python
await conn.refresh_gateways()
```

---

## 类型化 Resolve

如果有 actor 类定义，传入 `cls=` 参数可提前验证方法名：

```python
from pulsing.connect import Connect
from myapp.actors import Calculator

conn = await Connect.to("10.0.1.1:8080")
calc = await conn.resolve("services/calc", cls=Calculator)

# 拼写错误会立即抛出 AttributeError，而不是远程错误
calc.mulitply(6, 7)  # AttributeError: No method 'mulitply'
```

---

## 错误处理

### Actor 错误

远程 actor 内部抛出的错误会传播到调用方：

```python
from pulsing.exceptions import PulsingActorError

try:
    await calc.will_fail()
except PulsingActorError as e:
    print(f"Actor 错误: {e}")
```

### 流式错误

如果 actor 在流中途抛出异常，已接收的数据仍然可用：

```python
items = []
try:
    async for item in svc.partial_stream(10):
        items.append(item)
except PulsingActorError as e:
    print(f"流式错误，已收到 {len(items)} 项: {e}")
```

### Resolve 错误

解析不存在的 actor 会抛出异常：

```python
try:
    await conn.resolve("services/nonexistent")
except Exception as e:
    print(f"未找到: {e}")
```

---

## 集群内 vs 集群外

| 能力 | 集群内 (`pul.init()`) | 集群外 (`Connect.to()`) |
|------|:---:|:---:|
| `resolve` 命名 actor | ✅ | ✅ |
| 同步方法调用 | ✅ | ✅ |
| 异步方法调用 | ✅ | ✅ |
| 流式传输 | ✅ | ✅ |
| `spawn` actor（远程） | ✅ | ✅ |
| `spawn` actor（本地） | ✅ | ❌ |
| 被其他 actor 发现 | ✅ | ❌ |
| Gossip 成员协议 | ✅ | ❌ |
| 占用端口 | ✅ | ❌ |

**同样的 actor 代码，同样的调用语法。** 只有获取 proxy 的方式不同。

---

## 最佳实践

1. **关闭连接** — 用完后务必调用 `await conn.close()` 释放资源。
2. **使用多 gateway** — 生产环境传入多个地址以实现高可用。
3. **处理错误** — 在 try-except 中包裹远程调用以增强健壮性。
4. **使用类型化 resolve** — 传入 `cls=` 参数可提前发现拼写错误。
5. **区分流式与非流式** — 单个结果用 `await`，真正的流用 `async for`。

---

## 下一步

- [远程 Actor](remote_actors.md) — 集群内 actor 通信
- [通信范式](communication_patterns.md) — 同步、异步、流式模式
- [设计文档：Out-Cluster Connect](../design/out-cluster-connect.md) — 架构与协议细节
