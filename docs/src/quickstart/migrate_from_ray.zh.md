# 教程：从 Ray 迁移

**5 分钟**内用 Pulsing 替换 Ray。一行导入改动，零外部依赖。

---

## 为什么迁移？

| | Ray | Pulsing |
|---|-----|---------|
| **依赖** | Ray 集群、Redis、GCS | 无 |
| **启动时间** | 秒级 | 毫秒级 |
| **内存开销** | 高 | 低 |
| **Actor 模型** | 带状态的远程对象 | 经典模型（邮箱、FIFO） |
| **流式消息** | 手动实现 | 原生支持 |

---

## 步骤 1：修改导入

```python
# 之前 (Ray)
import ray

# 之后 (Pulsing)
from pulsing.compat import ray
```

**完成了。** 现有代码直接可用。

---

## 步骤 2：运行代码

### 之前 (Ray)

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
print(ray.get(counter.inc.remote()))  # 1
print(ray.get(counter.inc.remote()))  # 2

ray.shutdown()
```

### 之后 (Pulsing)

```python
from pulsing.compat import ray  # ← 只改了这一行

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1
        return self.value

counter = Counter.remote()
print(ray.get(counter.inc.remote()))  # 1
print(ray.get(counter.inc.remote()))  # 2

ray.shutdown()
```

---

## 支持的 API

| API | 状态 |
|-----|------|
| `ray.init()` | ✅ |
| `ray.shutdown()` | ✅ |
| `@ray.remote` (类) | ✅ |
| `@ray.remote` (函数) | ✅ |
| `ray.get()` | ✅ |
| `ray.put()` | ✅ |
| `ray.wait()` | ✅ |
| `ActorClass.remote()` | ✅ |
| `actor.method.remote()` | ✅ |

---

## 分布式模式

Ray 需要集群。Pulsing 只需要 `--addr` 和 `--seeds`：

### 节点 1（种子）

```python
from pulsing.compat import ray

ray.init(address="0.0.0.0:8000")

@ray.remote
class Worker:
    def process(self, data):
        return f"processed: {data}"

worker = Worker.remote()
# 保持运行...
```

### 节点 2（加入）

```python
from pulsing.compat import ray

ray.init(address="0.0.0.0:8001", seeds=["192.168.1.1:8000"])

# 查找远程 Actor
worker = ray.get_actor("Worker")
result = ray.get(worker.process.remote("hello"))
```

---

## 原生异步 API（可选）

新代码建议使用原生异步 API：

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
    counter = await Counter.spawn()
    print(await counter.inc())  # 1
    await pul.shutdown()
```

**优势：**

- 更简洁的 `async/await` 语法
- 无需 `ray.get()` 样板代码
- IDE 自动补全正常工作
- 可使用流式消息

---

## 限制

Ray 兼容 API 不支持：

- Ray Serve
- Ray Tune
- Ray Data
- Object Store（大对象）
- Placement Groups

这些功能请继续使用 Ray。Pulsing 专注于 Actor 模型。

---

## 下一步

- [指南：Actor](../guide/actors.zh.md) — 理解 Actor 模型
- [指南：远程 Actor](../guide/remote_actors.zh.md) — 集群设置
- [教程：LLM 推理](llm_inference.zh.md) — 构建推理服务
