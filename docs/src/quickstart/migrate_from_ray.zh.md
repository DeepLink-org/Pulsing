# 教程：Ray + Pulsing

用 Pulsing 作为 Ray Actor 的通信骨干——增加流式、Actor 发现和跨集群调用能力，无需替换 Ray。

---

## 两种使用方式

1. **桥接模式** — 保留 Ray Actor，通过 `pul.mount()` 接入 Pulsing 通信
2. **独立模式** — 直接使用 Pulsing 原生 API（适合新项目或完全迁移）

---

## 桥接模式：为 Ray Actor 增加 Pulsing 通信

最简单的路径——Ray 负责调度，Pulsing 负责通信：

```python
import ray
import pulsing as pul

@ray.remote
class Worker:
    def __init__(self, name):
        pul.mount(self, name=name)  # 一行代码：接入 Pulsing 网络

    async def call_peer(self, peer_name, msg):
        proxy = (await pul.resolve(peer_name, timeout=30)).as_any()
        return await proxy.greet(msg)  # 跨进程 Pulsing 调用

    async def greet(self, msg):
        return f"hello: {msg}"

ray.init()
workers = [Worker.remote(f"w{i}") for i in range(3)]
ray.get(workers[0].call_peer.remote("w1", "hi"))  # => "hello: hi"
pul.cleanup_ray()
```

**你获得的能力：** Ray 处理进程调度和资源管理。Pulsing 增加流式、命名 Actor 发现和直接的 Actor 间通信——不经过 Ray 的对象存储。

---

## 独立模式：Pulsing 原生 API

适合新项目或需要 Pulsing 完整特性的场景：

### API 对照表（Ray -> Pulsing）

| Ray | Pulsing |
|---|---|
| `ray.init()` | `await pul.init()` |
| `ray.shutdown()` | `await pul.shutdown()` |
| `@ray.remote` | `@pul.remote` |
| `Actor.remote(args...)` | `await Actor.spawn(args...)` |
| `ray.get(actor.method.remote(args...))` | `await actor.method(args...)` |
| `ray.get_actor(name)` | `await Actor.resolve(name)` 或 `await pul.resolve(name)` |

### 最小示例

**Ray：**

```python
import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    def inc(self):
        self.value += 1
        return self.value

counter = Counter.remote()
print(ray.get(counter.inc.remote()))
ray.shutdown()
```

**Pulsing：**

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self):
        self.value = 0
    def inc(self):
        self.value += 1
        return self.value

async def main():
    await pul.init()
    counter = await Counter.spawn(name="counter")
    print(await counter.inc())
    await pul.shutdown()
```

**关键差异：**

| 方面 | Ray | Pulsing |
|------|-----|---------|
| 创建 Actor | `Counter.remote()` | `await Counter.spawn()` — 原生 async |
| 调用方法 | `ray.get(counter.inc.remote())` | `await counter.inc()` — 直接 await |
| 按名获取 | `ray.get_actor("counter")` | `await Counter.resolve("counter")` — 带类型代理 |
| 流式 | 非内置 | 原生 `async for chunk in actor.stream()` |
| 发现 | 需要 GCS | 内置 gossip，零外部依赖 |

心智模型一致（远程类、spawn、方法调用）。Pulsing 增加了原生 async、流式和自包含集群能力。

---

## 分布式模式对照

### 节点 1（种子）

```python
import pulsing as pul

@pul.remote
class Worker:
    def process(self, data: str) -> str:
        return f"processed: {data}"

await pul.init(addr="0.0.0.0:8000")
await Worker.spawn(name="worker")
```

### 节点 2（加入 + 解析）

```python
import pulsing as pul

await pul.init(addr="0.0.0.0:8001", seeds=["192.168.1.1:8000"])
worker = await Worker.resolve("worker")
result = await worker.process("hello")
```

---

## 说明

- 优先使用 typed proxy：`await Class.resolve(name)`。
- 若只有运行时名称：`ref = await pul.resolve(name)`，再使用 `ref.as_type(Class)` / `ref.as_any()`。

---

## 下一步

- [指南：Actor](../guide/actors.zh.md) — 理解 Actor 模型
- [指南：远程 Actor](../guide/remote_actors.zh.md) — 集群设置
- [教程：LLM 推理](llm_inference.zh.md) — 构建推理服务
