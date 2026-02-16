# Quick Start

Get from zero to a **distributed Actor** in about **10 minutes** with three steps: your first Actor, a stateful Actor, then the same code across two nodes.

---

## Installation

```bash
pip install pulsing
```

---

## 1. Your First Actor (~2 minutes)

Define a class, add `@pul.remote`, then spawn and call it.

```python
import asyncio
import pulsing as pul

@pul.remote
class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

async def main():
    await pul.init()
    greeter = await Greeter.spawn()
    print(await greeter.greet("World"))  # Hello, World!
    await pul.shutdown()

asyncio.run(main())
```

The `@pul.remote` decorator turns the class into a distributed Actor. `spawn()` creates an instance; method calls use normal `await`.

---

## 2. Stateful Actor (~3 minutes)

Actors hold state. Here, a counter keeps a value and exposes `inc` and `get`.

```python
import asyncio
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, value: int = 0):
        self.value = value

    def inc(self, n: int = 1) -> int:
        self.value += n
        return self.value

    def get(self) -> int:
        return self.value

async def main():
    await pul.init()
    counter = await Counter.spawn(value=0)
    print(await counter.inc())   # 1
    print(await counter.inc(2))  # 3
    print(await counter.get())   # 3
    await pul.shutdown()

asyncio.run(main())
```

Same idea: one Actor instance, private state, messages via method calls. No shared memory, no locks.

---

## 3. Distributed: Same Code, Two Nodes (~5 minutes)

Run the same Actor type on two processes. Only the **initialization** changes: bind an address on the first node, join with `seeds` on the second.

**Node 1 (seed):**

```python
import asyncio
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, value: int = 0):
        self.value = value
    def inc(self, n: int = 1) -> int:
        self.value += n
        return self.value

async def main():
    await pul.init(addr="0.0.0.0:8000")
    await Counter.spawn(value=0, name="counter")
    await asyncio.Event().wait()  # keep running

asyncio.run(main())
```

**Node 2 (join cluster, then resolve and call):**

```python
import asyncio
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, value: int = 0):
        self.value = value
    def inc(self, n: int = 1) -> int:
        self.value += n
        return self.value

async def main():
    await pul.init(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])
    counter = await Counter.resolve("counter")
    print(await counter.inc(10))  # 10 — same API, remote actor
    await pul.shutdown()

asyncio.run(main())
```

**What changed:** `init(addr=..., seeds=...)` and `Counter.resolve("counter")` instead of `spawn()`. The rest of your code stays the same — **location transparency**.

---

## Next: Choose Your Path

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **LLM Inference Service**

    ---

    Build a scalable inference backend with streaming and OpenAI-compatible API.

    [:octicons-arrow-right-24: ~10 minutes](llm_inference.md)

-   :material-account-group:{ .lg .middle } **Distributed Agents**

    ---

    Integrate with AutoGen and LangGraph to distribute agents across machines.

    [:octicons-arrow-right-24: ~10 minutes](agent.md)

-   :material-swap-horizontal:{ .lg .middle } **Use with Ray**

    ---

    Bridge Ray actors onto the Pulsing network with `pul.mount()`. Add streaming and discovery to your Ray cluster.

    [:octicons-arrow-right-24: ~5 minutes](migrate_from_ray.md)

</div>

---

## Go Deeper

| Goal | Link |
|------|------|
| Named actors and ask vs tell | [Actor Patterns](patterns.md) |
| Form a cluster (Gossip / Head / Ray) | [Cluster Setup](cluster_networking.md) |
| Actor basics and patterns | [Actor Guide](../guide/actors.md) |
| When to use ask / tell / streaming | [Communication Patterns](../guide/communication_patterns.md) |
| Cluster setup and resolve | [Remote Actors](../guide/remote_actors.md) |
| Operate and inspect | [Operations](../guide/operations.md) |
