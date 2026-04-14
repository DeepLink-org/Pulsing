# Python Examples

## Setup

```bash
pip install -e .
```

## Run

### 原生异步 API (`pulsing.actor`)

Pulsing 原生 API，简洁高效：

```bash
# @remote 装饰器 + await 模式
python examples/python/remote_actor_example.py

# 原生异步 API 详细示例
python examples/python/native_async_example.py
```

### 与 Ray 一起使用

在 Ray 中希望使用 Pulsing 做通信时，请用 **Bridge 模式**（保留 Ray 调度，用 `pul.mount()` 将现有对象挂到 Pulsing 网络），或参考文档 [Tutorial: Ray + Pulsing](../../docs/src/quickstart/migrate_from_ray.md)。Ray 风格兼容层（`pulsing.compat.ray`）已移除，推荐使用原生 `await pul.init()` + `@pul.remote` 或 Bridge 模式。

```bash
python examples/python/transfer_queue_ray_rollout.py  # Ray rollout -> 3 transfer_queue buckets -> 3 trainer ranks
```

### 基础示例

```bash
python examples/python/ping_pong.py        # Basic communication
python examples/python/message_patterns.py # RPC and streaming
python examples/python/named_actors.py     # Service discovery
python examples/python/cluster.py          # Multi-node (see --help)
python examples/python/subprocess_example.py            # Native subprocess-compatible API
USE_POLSING_SUBPROCESS=1 python examples/python/subprocess_example.py --resources  # Pulsing backend
```

同步包装器说明：

- `transfer_queue.get_client()`、`queue.sync()`、`pulsing.subprocess` 的 Pulsing 后端都依赖 Pulsing 自己维护的 event loop。
- 这些同步 API 应该从同步代码或另一个线程里调用；如果当前已经在 async / Pulsing event loop 线程里，请改用 async API，或者放到 `asyncio.to_thread(...)` 里。

## API 选择

| API | 风格 | 适用场景 |
|-----|------|----------|
| `import pulsing as pul` | 异步 (`async/await`) | 新项目、Ray Bridge、高性能需求 |

### 原生 API 示例

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, value=0):
        self.value = value
    def inc(self):
        self.value += 1
        return self.value

async def main():
    await pul.init()
    counter = await Counter.spawn(value=0)
    print(await counter.inc())  # 1
    await pul.shutdown()
```

### Ray Bridge 示例（在 Ray worker 内挂载 Pulsing）

```python
import ray
import pulsing as pul

@ray.remote
class Worker:
    def __init__(self, name):
        pul.mount(self, name=name)

    async def greet(self, msg):
        return f"hello: {msg}"

# 见 docs/src/quickstart/migrate_from_ray.md 完整示例
```
