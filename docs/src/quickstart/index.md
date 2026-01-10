# Getting Started

Get up and running with Pulsing quickly.

## Installation

### Prerequisites

- **Python 3.10+**
- **Rust toolchain** (for building native extensions)
- **Linux/macOS**

### From Source

```bash
git clone https://github.com/reiase/pulsing.git
cd pulsing

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and install
pip install maturin
maturin develop
```

### From PyPI

```bash
pip install pulsing
```

---

## What is an Actor?

An Actor is an isolated computational unit with private state that processes messages sequentially. Local and remote Actors use the same API.

```mermaid
graph LR
    A[Sender] -->|Message| B[Actor Mailbox]
    B --> C[Actor]
    C -->|Response| A

    style A fill:#6366F1,color:#fff
    style B fill:#818CF8,color:#fff
    style C fill:#818CF8,color:#fff
```

---

## Your First Actor (30 seconds)

### Option 1: Native Async API (Recommended)

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

### Option 2: Ray-Compatible API (Easy Migration)

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
print(ray.get(counter.inc.remote()))  # 2

ray.shutdown()
```

**Any Python object** can be a message—strings, dicts, lists, or custom classes.

---

## Stateful Actors

```python
class Counter(Actor):
    def __init__(self):
        self.value = 0

    async def receive(self, msg):
        if msg == "inc":
            self.value += 1
            return self.value
        if msg == "get":
            return self.value
```

---

## API Comparison

| API | Style | Best For |
|-----|-------|----------|
| `pulsing.actor` | Async (`await`) | New projects, performance |
| `pulsing.compat.ray` | Sync (Ray-style) | Ray migration, quick start |

### @remote Decorator (Native API)

```python
from pulsing.actor import init, shutdown, remote

@remote
class Counter:
    def __init__(self, initial=0):
        self.value = initial

    def inc(self, n=1):
        self.value += n
        return self.value

async def main():
    await init()
    counter = await Counter.spawn(initial=10)
    print(await counter.inc(5))   # 15
    await shutdown()
```

---

## Cluster Communication

Pulsing uses SWIM gossip protocol—no external services needed!

**Node 1 (Seed):**
```python
config = SystemConfig.with_addr("0.0.0.0:8000")
system = await create_actor_system(config)
await system.spawn("worker", MyActor(), public=True)
```

**Node 2 (Join):**
```python
config = SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["node1:8000"])
system = await create_actor_system(config)

worker = await system.resolve_named("worker")
result = await worker.ask("do_work")  # Same API!
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Actor** | Isolated unit with private state |
| **Message** | Any Python object |
| **@remote** | Native async decorator (via `pulsing.actor`) |
| **ray.remote** | Ray-compatible decorator (via `pulsing.compat.ray`) |
| **Cluster** | SWIM protocol auto-discovery |

---

## Next Steps

- [Actor Guide](../guide/actors.md) - Advanced patterns
- [Agent Frameworks](../agent/index.md) - AutoGen and LangGraph integration
- [Examples](../examples/index.md) - Real-world use cases
