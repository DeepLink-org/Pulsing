---
template: home.html
title: Pulsing - Backbone for Distributed AI Systems
description: "Pulsing: Backbone for distributed AI systems. Actor runtime. Streaming-first. Zero dependencies. Built-in discovery."
hide: toc
---

<!-- This content is hidden by the home.html template but indexed for search -->

# Pulsing

**Backbone for distributed AI systems.**

Actor runtime. Streaming-first. Zero dependencies. Built-in discovery.

A distributed actor runtime built in Rust, designed for Python. Connect AI agents and services across machines — no Redis, no etcd, no YAML.

## Why Pulsing?

<div class="grid cards" markdown>

-   :material-package-variant-closed:{ .lg .middle } **Zero Dependencies**

    ---

    Pure Rust + Tokio. No etcd, NATS, Redis, or Consul. Just `pip install pulsing`.

-   :material-lightning-bolt:{ .lg .middle } **Streaming-first**

    ---

    Native streaming support built for LLM token generation and real-time communication.

-   :material-radar:{ .lg .middle } **Built-in Discovery**

    ---

    SWIM/Gossip protocol for automatic node discovery and failure detection. No configuration needed.

-   :material-language-python:{ .lg .middle } **Built in Rust, Designed for Python**

    ---

    Full async Python API via PyO3. `@remote` decorator turns any class into a distributed Actor.

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

-   :material-swap-horizontal:{ .lg .middle } **Works with Ray**

    ---

    Use Pulsing as the communication layer for Ray actors. Streaming, discovery, and cross-cluster calls — built in.

    [:octicons-arrow-right-24: Ray + Pulsing](quickstart/migrate_from_ray.md)

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

[:octicons-arrow-right-24: Quick Start](quickstart/index.md){ .md-button }

---

## Learn More

| Goal | Link |
|------|------|
| What is Pulsing / who is it for? | [Overview](overview.md) |
| Understand the Actor model | [Actor Basics](guide/actors.md) |
| Build a cluster | [Remote Actors](guide/remote_actors.md) |
| Operate your system | [CLI Operations](guide/operations.md) |
| Architecture and design | [Architecture & Design](design/architecture.md) |
| API details | [API Overview](api/overview.md) |
| Full API contract | [Complete Reference](api_reference.md) |

---

## Community

- [GitHub Repository](https://github.com/DeepLink-org/pulsing)
- [Issue Tracker](https://github.com/DeepLink-org/pulsing/issues)
- [Discussions](https://github.com/DeepLink-org/pulsing/discussions)
