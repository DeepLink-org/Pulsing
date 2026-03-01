# Pulsing API Reference for LLMs

## Overview

**Pulsing: Backbone for distributed AI systems.**

Pulsing is a distributed actor runtime built in Rust, designed for Python. Actor runtime. Streaming-first. Zero dependencies. Built-in discovery. Connect AI agents and services across machines — no Redis, no etcd, no YAML.

## Quick Start

```python
import pulsing as pul

await pul.init()

@pul.remote
class Counter:
    def __init__(self): self.value = 0
    def incr(self): self.value += 1; return self.value

# Create actor
counter = await Counter.spawn(name="counter")
print(await counter.incr())  # 1

# Resolve from another process / node
counter2 = await Counter.resolve("counter")
print(await counter2.incr())  # 2

await pul.shutdown()
```

## Python API

### 1. Init & Lifecycle

```python
import pulsing as pul

await pul.init(
    addr: str | None = None,           # Bind address; None = standalone
    *,
    seeds: list[str] | None = None,    # Seed nodes (gossip mode)
    passphrase: str | None = None,     # TLS passphrase
    head_addr: str | None = None,      # Worker mode: head node address
    is_head_node: bool = False,        # Head mode (mutually exclusive with head_addr)
)
await pul.shutdown()

pul.get_system() -> ActorSystem       # Get global system (raises if not init)
pul.is_initialized() -> bool
```

### 2. Define & Use Actors

```python
@pul.remote
class Counter:
    def __init__(self, init=0): self.value = init

    def incr(self):                          # sync: blocks actor, sequential
        self.value += 1; return self.value

    async def fetch_and_add(self, url):      # async: non-blocking during await
        data = await http_get(url)
        self.value += data; return self.value

    async def stream(self, n):               # generator → auto streaming response
        for i in range(n): yield f"chunk_{i}"

# ── Spawn ──
counter = await Counter.spawn(name="counter")
counter = await Counter.spawn(name="c2", placement="remote")  # remote node
counter = await Counter.spawn(name="c3", placement=3)         # specific node_id
counter = await Counter.spawn(system=system, name="c4")       # explicit ActorSystem

# ── Call ──
result = await counter.incr()
async for chunk in counter.stream(10): print(chunk)

# ── Resolve (cross-process / cross-node) ──
proxy = await pul.resolve("counter", cls=Counter, timeout=30)      # typed proxy
proxy = await pul.resolve("counter", timeout=30)                   # untyped proxy (any method)
proxy = await Counter.resolve("counter")                           # via ActorClass (also typed)
```

### 3. Supervision & Restart

```python
@pul.remote(
    restart_policy="on_failure",  # "never" | "on_failure" / "on-failure" | "always"
    max_restarts=3, min_backoff=0.1, max_backoff=30.0,
)
class ResilientWorker:
    def process(self, data): return heavy_computation(data)
```

### 4. Queue (distributed data pipeline)

```python
writer = await pul.queue.write("my_queue", bucket_column="id", num_buckets=4,
                               batch_size=100, backend="memory")  # -> Queue
await writer.put({"id": "u1", "data": "hello"})
await writer.flush()

reader = await pul.queue.read("my_queue", rank=0, world_size=4)  # -> QueueReader
records = await reader.get(limit=100, wait=False)
```

### 5. Topic (pub/sub)

```python
writer = await pul.topic.write("events")
await writer.publish({"type": "login", "user": "alice"})

reader = await pul.topic.read("events")

@reader.on_message
async def handle(msg): print(msg)

await reader.start()
```

### 6. Cluster & Integrations

```python
# ── Auto cluster formation (Ray / torchrun) ──
pul.bootstrap(*, ray=True, torchrun=True, on_ready=None, wait_timeout=None) -> bool | None

# ── Per-worker init ──
pul.init_inside_ray()        # join cluster from Ray worker
pul.init_inside_torchrun()   # join cluster from torchrun worker
pul.cleanup_ray()            # call before ray.shutdown()

# Typical Ray driver:
from pulsing.integrations.ray import init_in_ray
ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})
pul.bootstrap(ray=True, wait_timeout=30)
```

**mount / unmount** — register any object as a Pulsing actor (useful in Ray actors):

```python
pul.mount(instance, *, name: str, public: bool = True)  # sync, can be called in __init__
pul.unmount(name: str)

# Example
@ray.remote
class Worker:
    def __init__(self, name):
        pul.mount(self, name=name)

    async def call_peer(self, peer, msg):
        return await (await pul.resolve(peer, timeout=30)).greet(msg)
```

### 7. Error Handling

```python
from pulsing import (
    PulsingError,              # Base
    PulsingRuntimeError,       # Framework-level (actor not found, transport, cluster)
    PulsingActorError,         # User actor execution errors
    PulsingBusinessError,      # Business logic (code, message, details)
    PulsingSystemError,        # Internal (error, recoverable)
    PulsingTimeoutError,       # Timeout (operation, duration_ms)
    PulsingUnsupportedError,   # Unsupported operation
)
```

### 8. Advanced: Low-level API & Actor System

The global API is backed by an `ActorSystem`. Create one explicitly for finer control. Low-level APIs operate on `ActorRef` and require a `receive(self, msg)` method.

**Message / StreamMessage** are not exported from top-level `pulsing`. For low-level `receive()` or streaming, use:

```python
from pulsing.core.messaging import Message, StreamMessage
```

**Wire protocol:** Python–runtime call/response uses a flat format: `__call__` / `__async__`, `args`, `kwargs` for requests; `__result__` / `__error__` for responses. The legacy namespaced format (`__pulsing_proto__` / `user_data`) is no longer used.

```python
system = await pul.actor_system(addr=..., seeds=..., passphrase=...)

# Low-level spawn (actor must have receive method)
ref = await pul.spawn(actor, *, name=None, public=False,
                      restart_policy="never", max_restarts=3,
                      min_backoff=0.1, max_backoff=30.0) -> ActorRef

# Message passing on ActorRef
response = await ref.ask(request)   # request-response
await ref.tell(msg)                 # fire-and-forget

# Resolve / refer
ref = await pul.refer(actor_id)     # by ActorId
proxy = await pul.resolve(name, *, cls=None, node_id=None, timeout=None)

# Queue / Topic on explicit system
writer = await system.queue.write("q"); reader = await system.queue.read("q")
writer = await system.topic.write("t"); reader = await system.topic.read("t")
```

**Actor base class** (for low-level use):

```python
from pulsing.core import Actor, ActorId

class MyActor(Actor):
    def on_start(self, actor_id: ActorId): ...   # lifecycle hook
    def on_stop(self): ...
    def metadata(self) -> dict[str, str]: ...     # diagnostics
    async def receive(self, msg): return msg       # sync or async, auto-detected
```

**Zerocopy** — optional fast path bypassing pickle for buffer objects (from `pulsing.core` or `pulsing._core`):

```python
from pulsing.core import ZeroCopyDescriptor

class MyTensor:
    def __zerocopy__(self, ctx):
        return ZeroCopyDescriptor(
            buffers=[memoryview(self.buf)], dtype="float32",
            shape=[1024], strides=[4], transport="inline",
        )
# Missing __zerocopy__ → automatic pickle fallback
# Large buffers (>64KB) auto-use stream transfer (descriptor + chunked data)
```

Env vars: `PULSING_ZEROCOPY` (`auto`/`off`/`force`), `PULSING_ZEROCOPY_STREAM_THRESHOLD` (default 65536), `PULSING_ZEROCOPY_CHUNK_BYTES` (default 1048576).

---

## Rust API

### Quick Start

```rust
use pulsing_actor::prelude::*;

#[derive(Serialize, Deserialize)] struct Ping(i32);
#[derive(Serialize, Deserialize)] struct Pong(i32);
struct Echo;

#[async_trait]
impl Actor for Echo {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let Ping(x) = msg.unpack()?;
        Message::pack(&Pong(x))
    }
}

let system = ActorSystem::builder().build().await?;
let echo = system.spawn_named("services/echo", Echo).await?;
let Pong(x): Pong = echo.ask(Ping(1)).await?;
system.shutdown().await?;
```

### Spawn & Resolve

```rust
// Simple
system.spawn(actor).await?;                  // anonymous
system.spawn_named(name, actor).await?;      // named (resolvable)

// Builder pattern
system.spawning()
    .name("services/counter")
    .supervision(SupervisionSpec::on_failure().max_restarts(3))
    .mailbox_capacity(256)
    .spawn(actor).await?;

// Named + factory (restartable supervision)
system.spawn_named_factory(name, || Ok(Service::new()), options).await?;

// Resolve
system.resolve(name).await?;                 // one-shot
system.resolving().lazy(name)?;              // lazy + auto-refresh (~5s TTL)
system.resolving().node(id).policy(RoundRobinPolicy::new()).resolve(name).await?;
```

### Operations

```rust
system.node_id(); system.addr(); system.members().await;
system.all_named_actors().await; system.stop(name).await?; system.shutdown().await?;
```

### Behavior (Type-safe, Akka Typed style)

```rust
fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n, _ctx| { *count += n; BehaviorAction::Same })
}
let c = system.spawn_named("actors/counter", counter(0)).await?;
```

### Key Conventions

- **Message encoding**: `Message::pack(&T)` (bincode); cross-version: `Message::single("TypeV1", bytes)`.
- **Streaming**: Return `Message::Stream`, cancellation is best-effort.
- **Supervision**: Only `spawn_named_factory` supports restart; anonymous actors do not.
