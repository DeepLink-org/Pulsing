# 教程：从 Ray 迁移

将 Ray Actor 代码迁移到 Pulsing 原生异步 API。

---

## 为什么这篇迁移说明改了

当前项目不再推荐 Ray 兼容层（`pulsing.compat.ray`）。
请直接使用 Pulsing 主 API：

- `import pulsing as pul`
- `@pul.remote`
- `await pul.init()` / `await pul.shutdown()`
- `await Class.spawn()` / `await Class.resolve()`

---

## API 对照表（Ray -> Pulsing）

| Ray | Pulsing |
|---|---|
| `ray.init()` | `await pul.init()` |
| `ray.shutdown()` | `await pul.shutdown()` |
| `@ray.remote` | `@pul.remote` |
| `Actor.remote(args...)` | `await Actor.spawn(args...)` |
| `ray.get(actor.method.remote(args...))` | `await actor.method(args...)` |
| `ray.get_actor(name)` | `await Actor.resolve(name)` 或 `await pul.resolve(name)` |

---

## 最小迁移示例

### 之前（Ray）

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

### 之后（Pulsing）

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
