# Actor 指南

本指南介绍 **Actor 模型** 概念和构建健壮分布式应用的模式。

!!! tip "前置要求"
    如果尚未完成 [快速开始](../quickstart/index.zh.md)，请先阅读。

---

## 什么是 Actor？

**Actor** 是并发和分布式系统中的基本计算单元。Actor 模型由 Carl Hewitt 于 1973 年提出，提供了一种构建以下系统的原则性方法：

- **并发**：多个 Actor 并行运行
- **分布式**：Actor 可以在不同机器上
- **容错**：故障被隔离

### 核心原则

```
┌─────────────────────────────────────────────────────────────┐
│                         Actor                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ 私有状态     │    │  消息邮箱    │    │  行为       │     │
│  │ (State)     │    │  (FIFO)     │    │  (Methods)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        ▲                  │                   │             │
│        │                  ▼                   ▼             │
│        └───────── 每次只处理一条消息 ──────────────────────│
└─────────────────────────────────────────────────────────────┘
```

| 原则 | 描述 |
|------|------|
| **隔离性** | 每个 Actor 有私有状态；无共享内存 |
| **消息传递** | Actor 仅通过异步消息通信 |
| **顺序处理** | 每次只处理一条消息（无内部锁） |
| **位置透明** | 本地和远程 Actor 使用相同 API |

### 为什么选择 Pulsing 而非 Ray？

Ray 的 "actor" 本质上是**带状态的远程对象** — 你调用它的方法，但没有正式的消息队列或投递语义。

Pulsing 遵循**经典 Actor 模型**（类似 Erlang/Akka）：

| 特性 | Pulsing | Ray |
|------|---------|-----|
| 消息队列（邮箱） | ✅ FIFO | ❌ 直接调用 |
| 顺序保证 | ✅ 每 Actor | ⚠️ 每方法 |
| 监督/重启 | ✅ 内置 | ❌ 手动 |
| 零外部依赖 | ✅ | ❌ （需 Ray 集群） |
| 流式消息 | ✅ 原生 | ❌ |

---

## 两种 API 风格

| API | 导入方式 | 风格 | 适用场景 |
|-----|---------|------|----------|
| **原生异步** | `from pulsing.actor import ...` | `async/await` | 新项目，追求极致性能 |
| **Ray 兼容** | `from pulsing.compat import ray` | 同步调用 | 从 Ray 迁移，快速原型 |

### 原生异步 API（推荐）

```python
from pulsing.actor import init, shutdown, remote

@remote
class Calculator:
    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def add(self, n: int) -> int:
        self.value += n
        return self.value

async def main():
    await init()
    calc = await Calculator.spawn(initial_value=100)
    result = await calc.add(50)  # 150
    await shutdown()
```

### Ray 兼容 API

```python
from pulsing.compat import ray

ray.init()

@ray.remote
class Calculator:
    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def add(self, n: int) -> int:
        self.value += n
        return self.value

calc = Calculator.remote(initial_value=100)
result = ray.get(calc.add.remote(50))  # 150
ray.shutdown()
```

**从 Ray 迁移** — 只需修改导入：

```python
# 之前:  import ray
# 之后:  from pulsing.compat import ray
```

---

## 消息模式

### Ask（请求-响应）

```python
result = await calc.add(10)
```

### Tell（即发即忘）

```python
await actor_ref.tell(Message.single("notify", b"event_data"))
```

### 流式消息

用于持续数据流（如 LLM token 生成）：

```python
from pulsing.actor import StreamMessage

@remote
class TokenGenerator:
    async def generate(self, prompt: str) -> Message:
        stream_msg, writer = StreamMessage.create("tokens")

        async def produce():
            for token in self.generate_tokens(prompt):
                await writer.write({"token": token})
            await writer.close()

        asyncio.create_task(produce())
        return stream_msg

# 消费流
response = await generator.generate("Hello")
async for chunk in response.stream_reader():
    print(chunk["token"], end="", flush=True)
```

---

## 监督（Actor 级重启）

Pulsing 支持 Actor 失败后自动重启：

```python
@remote(
    restart_policy="on_failure",  # "never" | "on_failure" | "always"
    max_restarts=3,
    min_backoff=1.0,
    max_backoff=60.0
)
class ReliableWorker:
    def process(self, data):
        # 如果崩溃，Actor 会自动重启
        return heavy_computation(data)
```

!!! note
    重启恢复 Actor 但**不恢复**内存状态。参阅 [可靠性指南](reliability.zh.md) 了解幂等模式。

---

## 进阶模式

### 1. 有状态 Actor

```python
@remote
class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"user_id": user_id, "data": {}}
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        return self.sessions.get(session_id)
```

### 2. Worker 池（轮询）

```python
@remote
class WorkerPool:
    def __init__(self, workers: list):
        self.workers = workers
        self.idx = 0

    async def submit(self, task: dict):
        worker = self.workers[self.idx]
        self.idx = (self.idx + 1) % len(self.workers)
        return await worker.process(task)
```

### 3. 流水线

```python
@remote
class PipelineStage:
    def __init__(self, next_stage=None):
        self.next_stage = next_stage

    async def process(self, data: dict) -> dict:
        result = await self.transform(data)
        if self.next_stage:
            return await self.next_stage.process(result)
        return result
```

### 4. LLM 推理服务

```python
@remote
class LLMService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    async def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 最佳实践

| ✅ 应该 | ❌ 不应该 |
|--------|----------|
| Actor 单一职责 | 在 Actor 间共享可变状态 |
| I/O 使用 async | 在方法中阻塞 |
| 优雅处理错误 | 忽略异常 |
| 在 `__init__` 初始化状态 | 使用全局变量 |

### 错误处理

```python
@remote
class ResilientActor:
    async def risky_operation(self, data: dict) -> dict:
        try:
            result = await self.process(data)
            return {"success": True, "result": result}
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"意外错误: {e}")
            raise
```

---

## 快速参考

### 常用操作

```python
# 创建
actor = await MyActor.spawn(param=10)

# 调用方法
result = await actor.method(arg)

# 使用 system handle
system = await create_actor_system(config)
actor = await system.spawn(MyActor(), "name", public=True)
remote_actor = await system.find("remote-name")
await system.stop("name")
await system.shutdown()
```

---

## 下一步

- [远程 Actor](remote_actors.zh.md) — 集群通信
- [可靠性](reliability.zh.md) — 幂等、重试、超时
- [运维操作](operations.zh.md) — CLI 检查工具
- [LLM 推理](../examples/llm_inference.zh.md) — 生产推理部署
