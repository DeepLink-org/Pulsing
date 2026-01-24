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

### Ray 兼容 API (`pulsing.compat.ray`)

一行代码从 Ray 迁移到 Pulsing：

```bash
# Ray 风格 API，同步接口
python examples/python/ray_compat_example.py
```

### 基础示例

```bash
python examples/python/ping_pong.py        # Basic communication
python examples/python/message_patterns.py # RPC and streaming
python examples/python/named_actors.py     # Service discovery
python examples/python/cluster.py          # Multi-node (see --help)
```

## API 选择

| API | 风格 | 适用场景 |
|-----|------|----------|
| `import pulsing as pul` | 异步 (`async/await`) | 新项目，高性能需求 |
| `from pulsing.compat import ray` | 同步 (Ray 风格) | Ray 迁移，快速上手 |

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

### Ray 兼容 API 示例

```python
from pulsing.compat import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self, value=0):
        self.value = value
    def inc(self):
        self.value += 1
        return self.value

counter = Counter.remote(value=0)
print(ray.get(counter.inc.remote()))  # 1
ray.shutdown()
```
