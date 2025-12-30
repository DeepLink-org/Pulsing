# 示例

本节包含演示如何将 Pulsing 用于各种用例的实际示例。

## 入门示例

### Hello World

最简单的 Pulsing 应用程序：

```python
import asyncio
from pulsing.actor import as_actor, create_actor_system, SystemConfig

@as_actor
class HelloActor:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

async def main():
    system = await create_actor_system(SystemConfig.standalone())
    hello = await HelloActor.local(system)
    print(await hello.greet("World"))
    await system.shutdown()

asyncio.run(main())
```

### 计数器 Actor

维护计数器的有状态 Actor：

```python
@as_actor
class Counter:
    def __init__(self, initial: int = 0):
        self.value = initial

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

    def decrement(self, n: int = 1) -> int:
        self.value -= n
        return self.value

    def get(self) -> int:
        return self.value

    def reset(self) -> int:
        self.value = 0
        return self.value
```

## 分布式示例

### Ping-Pong

两个 Actor 跨节点通信：

```python
@as_actor
class PingActor:
    def __init__(self, pong_ref=None):
        self.pong = pong_ref
        self.count = 0

    async def start_ping(self, times: int):
        for i in range(times):
            response = await self.pong.pong(i)
            self.count += 1
            print(f"收到: {response}")
        return self.count

@as_actor
class PongActor:
    def pong(self, n: int) -> str:
        return f"pong-{n}"


async def main():
    # 节点 1：启动 pong Actor
    system1 = await create_actor_system(
        SystemConfig.with_addr("0.0.0.0:8000")
    )
    pong = await PongActor.local(system1)
    await system1.register("pong", pong, public=True)

    # 节点 2：启动 ping Actor
    system2 = await create_actor_system(
        SystemConfig.with_addr("0.0.0.0:8001")
        .with_seeds(["127.0.0.1:8000"])
    )
    await asyncio.sleep(1.0)  # 等待集群同步

    pong_ref = await system2.find("pong")
    ping = await PingActor.local(system2, pong_ref=pong_ref)

    # 运行 ping-pong
    count = await ping.start_ping(10)
    print(f"完成 {count} 轮 ping-pong")

    await system1.shutdown()
    await system2.shutdown()
```

### 工作池

将工作分配给多个工作器：

```python
@as_actor
class Worker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.tasks_completed = 0

    async def process(self, task: dict) -> dict:
        # 模拟工作
        await asyncio.sleep(0.1)
        self.tasks_completed += 1
        return {
            "worker_id": self.worker_id,
            "task": task,
            "result": f"processed-{task['id']}"
        }

    def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "tasks_completed": self.tasks_completed
        }


@as_actor
class WorkerPool:
    def __init__(self):
        self.workers = []
        self.next_worker = 0

    async def initialize(self, system, num_workers: int):
        for i in range(num_workers):
            worker = await Worker.local(system, worker_id=i)
            self.workers.append(worker)

    async def submit(self, task: dict) -> dict:
        # 轮询分配
        worker = self.workers[self.next_worker]
        self.next_worker = (self.next_worker + 1) % len(self.workers)
        return await worker.process(task)

    async def submit_batch(self, tasks: list) -> list:
        # 并行处理所有任务
        futures = [self.submit(task) for task in tasks]
        return await asyncio.gather(*futures)
```

## LLM 推理示例

### 简单 LLM 服务

```python
@as_actor
class LLMService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.request_count = 0

    async def load_model(self):
        """加载模型。创建后调用一次。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """生成文本补全。"""
        self.request_count += 1

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_stats(self) -> dict:
        return {
            "model": self.model_name,
            "requests_served": self.request_count
        }
```

## 更多示例

- [Ping-Pong](ping_pong.md) - 基本 Actor 通信
- [分布式计数器](distributed_counter.md) - 跨节点共享状态
- [LLM 推理](llm_inference.md) - 构建推理服务

## 运行示例

大多数示例可以直接运行：

```bash
# 运行单个示例
python examples/hello_world.py

# 运行分布式示例（需要多个终端）
# 终端 1：
python examples/ping_pong.py --node seed --port 8000

# 终端 2：
python examples/ping_pong.py --node worker --port 8001 --seed 127.0.0.1:8000
```
