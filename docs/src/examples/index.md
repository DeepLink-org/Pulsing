# Examples

This section contains practical examples demonstrating how to use Pulsing for various use cases.

## Getting Started Examples

### Hello World

The simplest possible Pulsing application:

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

### Counter Actor

A stateful actor that maintains a counter:

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

## Distributed Examples

### Ping-Pong

Two actors communicating across nodes:

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
            print(f"Received: {response}")
        return self.count

@as_actor
class PongActor:
    def pong(self, n: int) -> str:
        return f"pong-{n}"


async def main():
    # Node 1: Start pong actor
    system1 = await create_actor_system(
        SystemConfig.with_addr("0.0.0.0:8000")
    )
    pong = await PongActor.local(system1)
    await system1.register("pong", pong, public=True)

    # Node 2: Start ping actor
    system2 = await create_actor_system(
        SystemConfig.with_addr("0.0.0.0:8001")
        .with_seeds(["127.0.0.1:8000"])
    )
    await asyncio.sleep(1.0)  # Wait for cluster sync

    pong_ref = await system2.find("pong")
    ping = await PingActor.local(system2, pong_ref=pong_ref)

    # Run ping-pong
    count = await ping.start_ping(10)
    print(f"Completed {count} ping-pong rounds")

    await system1.shutdown()
    await system2.shutdown()
```

### Worker Pool

Distributing work across multiple workers:

```python
@as_actor
class Worker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.tasks_completed = 0

    async def process(self, task: dict) -> dict:
        # Simulate work
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
        # Round-robin distribution
        worker = self.workers[self.next_worker]
        self.next_worker = (self.next_worker + 1) % len(self.workers)
        return await worker.process(task)

    async def submit_batch(self, tasks: list) -> list:
        # Process all tasks in parallel
        futures = [self.submit(task) for task in tasks]
        return await asyncio.gather(*futures)

    async def get_all_stats(self) -> list:
        stats = await asyncio.gather(*[
            w.get_stats() for w in self.workers
        ])
        return stats
```

## LLM Inference Examples

### Simple LLM Service

```python
@as_actor
class LLMService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.request_count = 0

    async def load_model(self):
        """Load the model. Call once after creation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text completion."""
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

### Load-Balanced LLM Cluster

```python
@as_actor
class LLMRouter:
    """Routes requests to LLM workers with load balancing."""

    def __init__(self):
        self.workers = []
        self.request_counts = []

    async def add_worker(self, worker_ref):
        self.workers.append(worker_ref)
        self.request_counts.append(0)

    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        # Find worker with least requests
        min_idx = self.request_counts.index(min(self.request_counts))
        worker = self.workers[min_idx]
        self.request_counts[min_idx] += 1

        try:
            result = await worker.generate(prompt, max_tokens)
            return result
        finally:
            self.request_counts[min_idx] -= 1

    def get_load(self) -> list:
        return list(zip(range(len(self.workers)), self.request_counts))
```

## More Examples

- [Ping-Pong](ping_pong.md) - Basic actor communication
- [Distributed Counter](distributed_counter.md) - Shared state across nodes
- [LLM Inference](llm_inference.md) - Building inference services

## Running Examples

Most examples can be run directly:

```bash
# Run a single example
python examples/hello_world.py

# Run distributed examples (need multiple terminals)
# Terminal 1:
python examples/ping_pong.py --node seed --port 8000

# Terminal 2:
python examples/ping_pong.py --node worker --port 8001 --seed 127.0.0.1:8000
```
