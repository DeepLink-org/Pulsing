---
template: home.html
title: Pulsing - Lightweight Distributed Actor Framework for AI
description: Pulsing is a distributed actor framework that provides a communication backbone for building distributed systems, with specialized support for AI applications.
hide: toc
---

<!-- This content is hidden by the home.html template but indexed for search -->

# Pulsing

A **distributed actor framework** that provides a communication backbone for building distributed systems, with specialized support for AI applications.

## Why Pulsing?

<div class="grid cards" markdown>

-   :material-package-variant-closed:{ .lg .middle } **Zero External Dependencies**

    ---

    Pure Rust + Tokio. No etcd, NATS, Redis, or Consul required.

-   :material-radar:{ .lg .middle } **Built-in Cluster Discovery**

    ---

    SWIM/Gossip protocol for automatic node discovery and failure detection.

-   :material-lightning-bolt:{ .lg .middle } **High Performance**

    ---

    Async runtime with HTTP/2 transport and native streaming support.

-   :material-language-python:{ .lg .middle } **Python First**

    ---

    Full Python API via PyO3. `@remote` decorator turns any class into an Actor.

</div>

---

## What Can You Build?

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **LLM Inference Services**

    ---

    Scalable inference backends with streaming token generation. OpenAI-compatible API out of the box.

    [:octicons-arrow-right-24: LLM Inference](quickstart/llm_inference.md)

-   :material-account-group:{ .lg .middle } **Distributed Agents**

    ---

    Native integration with AutoGen and LangGraph. Distribute your agents across machines.

    [:octicons-arrow-right-24: Distributed Agents](quickstart/agent.md)

-   :material-swap-horizontal:{ .lg .middle } **Replace Ray**

    ---

    Drop-in compatible API. Migrate from Ray with one import change.

    [:octicons-arrow-right-24: Migrate from Ray](quickstart/migrate_from_ray.md)

</div>

---

## Quick Start

```bash
pip install pulsing
```

```python
import asyncio
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
    print(await counter.inc())  # 2
    await pul.shutdown()

asyncio.run(main())
```

[:octicons-arrow-right-24: Getting Started](quickstart/index.md){ .md-button }

---

## Learn More

| Goal | Link |
|------|------|
| Understand the Actor model | [Guide: Actors](guide/actors.md) |
| Build a cluster | [Guide: Remote Actors](guide/remote_actors.md) |
| Operate your system | [Guide: CLI Operations](guide/operations.md) |
| Deep dive into design | [Design Documents](design/architecture.md) |
| API details | [API Reference](api/overview.md) |

---

## Community

- [GitHub Repository](https://github.com/DeepLink-org/pulsing)
- [Issue Tracker](https://github.com/DeepLink-org/pulsing/issues)
- [Discussions](https://github.com/DeepLink-org/pulsing/discussions)
