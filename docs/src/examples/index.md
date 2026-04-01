# Examples

This section contains practical examples demonstrating how to use Pulsing for various use cases.

## Getting Started Examples

### Hello World

The simplest possible Pulsing application:

```python
import asyncio
import pulsing as pul

@pul.remote
class HelloActor:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

async def main():
    await pul.init()
    hello = await HelloActor.spawn()
    print(await hello.greet("World"))
    await pul.shutdown()

asyncio.run(main())
```

### Counter Actor

A stateful actor that maintains a counter:

```python
@pul.remote
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
import pulsing as pul

@pul.remote
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

@pul.remote
class PongActor:
    def pong(self, n: int) -> str:
        return f"pong-{n}"


async def main():
    # Node 1: Start pong actor
    await init(addr="0.0.0.0:8000")
    pong = await PongActor.spawn(name="pong")

    # Node 2: Would run on another machine
    # await init(addr="0.0.0.0:8001", seeds=["node1:8000"])
    # pong_ref = await PongActor.resolve("pong")
    # ping = await PingActor.spawn(pong_ref=pong_ref)
    # await ping.start_ping(10)

    await shutdown()
```

### Worker Pool

Distributing work across multiple workers:

```python
@pul.remote
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


@pul.remote
class WorkerPool:
    def __init__(self):
        self.workers = []
        self.next_worker = 0

    async def initialize(self, num_workers: int):
        for i in range(num_workers):
            worker = await Worker.spawn(worker_id=i)
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

### Subprocess

Run shell commands with a stdlib-compatible API:

```python
import pulsing.subprocess as subprocess

result = subprocess.run(["echo", "hello"], capture_output=True, text=True)
print(result.stdout.strip())

remote = subprocess.run(
    ["hostname"],
    capture_output=True,
    text=True,
    resources={"num_cpus": 1},
)
print(remote.stdout.strip())
```

Run the full example:

```bash
python examples/python/subprocess_example.py
python examples/python/subprocess_example.py --resources
USE_POLSING_SUBPROCESS=1 python examples/python/subprocess_example.py --resources
```

## LLM Inference Examples

### Simple LLM Service

```python
@pul.remote
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
@pul.remote
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

## Agent Framework Examples

Pulsing integrates with popular agent frameworks. See [Agent Integration](../agent/index.md) for details.

### AutoGen

Use `PulsingRuntime` as a drop-in replacement for AutoGen's runtime:

```python
from pulsing.integrations.autogen import PulsingRuntime

runtime = PulsingRuntime(addr="0.0.0.0:8000")
await runtime.start()
await runtime.register_factory("agent", lambda: MyAgent())
await runtime.send_message("Hello", AgentId("agent", "default"))
```

Run the example:
```bash
cd examples/agent/autogen && ./run_distributed.sh
```

### LangGraph

Use `with_pulsing()` to enable distributed execution:

```python
from pulsing.integrations.langgraph import with_pulsing

app = graph.compile()
distributed_app = with_pulsing(
    app,
    node_mapping={"llm": "langgraph_node_llm"},
    seeds=["gpu-server:8001"],
)
await distributed_app.ainvoke(inputs)
```

Run the example:
```bash
cd examples/agent/langgraph && ./run_distributed.sh
```

## More Examples

- [Ping-Pong](ping_pong.md) - Basic actor communication
- [Distributed Counter](distributed_counter.md) - Shared state across nodes
- [Subprocess](subprocess.md) - Subprocess-compatible command execution
- [LLM Inference](llm_inference.md) - Building inference services
- [AutoGen Integration](../agent/autogen.md) - Distributed AutoGen agents
- [LangGraph Integration](../agent/langgraph.md) - Distributed LangGraph workflows

## Running Examples

Most examples can be run directly:

```bash
# Run a single example
python examples/hello_world.py
python examples/python/subprocess_example.py

# Run distributed examples (need multiple terminals)
# Terminal 1:
python examples/ping_pong.py --node seed --port 8000

# Terminal 2:
python examples/ping_pong.py --node worker --port 8001 --seed 127.0.0.1:8000
```
