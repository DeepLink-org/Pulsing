# Getting Started

Get Pulsing running in **5 minutes**.

## Installation

```bash
pip install pulsing
```

---

## Your First Actor

```python
import asyncio
from pulsing.actor import init, shutdown, remote

@remote
class Counter:
    def __init__(self, value=0):
        self.value = value

    def inc(self):
        self.value += 1
        return self.value

async def main():
    await init()
    counter = await Counter.spawn(value=0)
    print(await counter.inc())  # 1
    print(await counter.inc())  # 2
    await shutdown()

asyncio.run(main())
```

The `@remote` decorator turns any Python class into a distributed Actor.

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

-   :material-swap-horizontal:{ .lg .middle } **Migrate from Ray**

    ---

    Replace Ray with one import change. Zero external dependencies.

    [:octicons-arrow-right-24: ~5 minutes](migrate_from_ray.md)

</div>

---

## Go Deeper

| Goal | Link |
|------|------|
| Understand the Actor model | [Guide: Actors](../guide/actors.md) |
| Build a cluster | [Guide: Remote Actors](../guide/remote_actors.md) |
| Operate your system | [Guide: Operations](../guide/operations.md) |
