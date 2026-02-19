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

You must call `await pul.init()` before using `spawn`, `resolve`, or other APIs.

```python
import pulsing as pul

# ── Lifecycle ──

await pul.init(
    addr: str | None = None,
    *,
    seeds: list[str] | None = None,
    passphrase: str | None = None
)

await pul.shutdown()

# ── Define actor with @pul.remote ──

@pul.remote
class Counter:
    def __init__(self, init=0): self.value = init

    def incr(self):                     # sync method
        self.value += 1
        return self.value

    async def fetch_and_add(self, url):  # async method
        data = await http_get(url)
        self.value += data
        return self.value

# ── Create and call ──

counter = await Counter.spawn(name="counter")   # create actor, returns typed proxy
result = await counter.incr()                    # call method directly

# ── Resolve existing actor (e.g. from another process / node) ──
# Prefer typed proxy via Counter.resolve() when you know the actor type.
# Fall back to ref.as_any() when the remote type is unknown.

# 1. Typed proxy (recommended)
proxy = await Counter.resolve("counter")
result = await proxy.incr()

# 2. Typed proxy — manual bind
ref = await pul.resolve("counter", timeout=30)
proxy = ref.as_type(Counter)
result = await proxy.incr()

# 3. Untyped proxy — when remote type is unknown
ref = await pul.resolve("service_name")
proxy = ref.as_any()
result = await proxy.any_method(args)

```

### Ray Integration

`pul.mount` registers any Python object as a Pulsing actor, enabling tight integration between Ray actors and Pulsing.

**Running Pulsing in a Ray cluster:** Every process (driver and workers) must initialize Pulsing. Use `pulsing.ray.init_in_ray()` and pass it in `ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})` so workers call it on startup; the driver must call `init_in_ray()` once in code. See the `pulsing.ray` module for details.

```python
import pulsing as pul

# Mount object onto Pulsing network (sync, can be called in __init__)
pul.mount(
    instance: Any,            # Object to mount
    *,
    name: str,                # Pulsing name, used for resolve discovery
    public: bool = True,      # Whether discoverable by other cluster nodes
) -> None
# Internally:
#   1. Initialize Pulsing (if not yet initialized in this process)
#   2. Wrap instance as a Pulsing actor
#   3. Register on Pulsing network, gossip broadcasts the name

# Unmount (call when actor is destroyed)
pul.unmount(name: str) -> None

# Cleanup Pulsing state in Ray environment (call before ray.shutdown())
pul.cleanup_ray() -> None
```
Example: Ray handles process scheduling, Pulsing handles inter-actor communication.

```python
import ray, pulsing as pul

@ray.remote
class Worker:
    def __init__(self, name):
        pul.mount(self, name=name)              # One line to join Pulsing

    async def call_peer(self, peer_name, msg):
        proxy = (await pul.resolve(peer_name, timeout=30)).as_any()
        return await proxy.greet(msg)           # Cross-process Pulsing call

    async def greet(self, msg):
        return f"hello from {self.name}: {msg}"

ray.init()
workers = [Worker.remote(f"w{i}") for i in range(3)]
ray.get(workers[0].call_peer.remote("w1", "hi"))  # => "hello from w1: hi"
pul.cleanup_ray()
```

### Under the Hood: Actor System & Low-level APIs

The global API is backed by an `ActorSystem` instance. You can create one explicitly when you need multiple systems or finer control. The low-level `spawn`/`refer`/`resolve` APIs operate on `ActorRef` (not typed proxy) and require actors to implement a `receive(self, msg)` method.

```python
import pulsing as pul

# ── Explicit ActorSystem ──

system = await pul.actor_system(
    addr: str | None = None,
    *,
    seeds: list[str] | None = None,
    passphrase: str | None = None
) -> ActorSystem

await system.shutdown()

# ── Low-level spawn (actor must have receive method) ──

actorref = await pul.spawn(       # global system
    actor: Actor,
    *,
    name: str | None = None,
    public: bool = False,
    restart_policy: str = "never",
    max_restarts: int = 3,
    min_backoff: float = 0.1,
    max_backoff: float = 30.0
) -> ActorRef

actorref = await system.spawn(    # explicit system, same signature
    actor: Actor, ...
) -> ActorRef

# ── Low-level resolve / refer ──

actorref = await pul.refer(actorid: ActorId | str) -> ActorRef
actorref = await pul.resolve(name: str, *, node_id=None, timeout=None) -> ActorRef
actorref = await system.resolve(name: str, *, node_id=None) -> ActorRef

# ── ActorRef message passing ──

response = await actorref.ask(request: Any) -> Any
await actorref.tell(msg: Any) -> None

# ── @pul.remote with explicit system ──

counter = await Counter.local(system, name="counter")  # spawn on explicit system
result = await counter.incr()

# Queue / Topic on explicit system (same API as pul.queue / pul.topic)
writer = await system.queue.write("my_queue")
reader = await system.queue.read("my_queue")
writer = await system.topic.write("events")
reader = await system.topic.read("events")
```

### Actor Behavior

#### Basic Actor (using `receive` method)

```python
from pulsing.actor import Actor

class EchoActor(Actor):
    """receive method - sync or async, framework auto-detects"""

    # Option 1: Synchronous
    def receive(self, msg):
        return msg

    # Option 2: Asynchronous (use when you need await)
    async def receive(self, msg):
        result = await some_async_operation()
        return result

class FireAndForget(Actor):
    """No return value (suitable for tell calls)"""
    def receive(self, msg):
        print(f"Received: {msg}")
        # No return value
```

**Note:** `receive` can be `def` or `async def`, Pulsing auto-detects and handles both correctly.
Only use `async def` when the method body needs to `await` other coroutines.

#### @pul.remote Decorator (Recommended)

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, init=0):
        self.value = init

    # Sync method - blocks actor, requests execute sequentially
    # Best for: fast computation, state mutation
    def incr(self):
        self.value += 1
        return self.value

    # Async method - non-blocking, can handle other requests during await
    # Best for: IO-bound operations (network, database)
    async def fetch_and_add(self, url):
        data = await http_get(url)  # Other requests served during await
        self.value += data
        return self.value

    # No return value - suitable for tell() calls
    def reset(self):
        self.value = 0

# Sync vs async concurrency behavior:
# - def method():       Blocks actor, requests queued sequentially
# - async def method(): Non-blocking, concurrent during await

# Usage
counter = await Counter.spawn(name="counter")
result = await counter.incr()      # ask mode, waits for return
await counter.reset()              # No return value, but still waits for completion
```

#### Message Passing Patterns

```python
# ask - send message and wait for response
response = await actorref.ask({"action": "get"})

# tell - send message, don't wait (fire-and-forget)
await actorref.tell({"action": "log", "data": "hello"})
```

#### Optional Zerocopy Descriptor Protocol

Pulsing supports an optional zerocopy fast path to bypass pickle serialization for eligible
Python objects. If the object does not provide the protocol, Pulsing falls back to existing
pickle-based transport automatically.

```python
from pulsing.core import ZeroCopyDescriptor

class MyTensorLike:
    def __zerocopy__(self, ctx):
        return ZeroCopyDescriptor(
            buffers=[memoryview(self.buffer)],
            dtype="float32",
            shape=[1024],
            strides=[4],
            transport="inline",   # e.g. inline/shm
            checksum=None,        # optional
            version=1,
        )
```

Rules:

- `__zerocopy__(ctx)` is optional; missing protocol means fallback to pickle.
- Descriptor is the single source of truth (no separate `__metadata__`).
- Zerocopy is an optimization path for reduced serialization and buffer copies.
- `buffers` should provide contiguous Python buffer views (e.g. `memoryview`, tensor buffer, `bytearray`) to avoid extra Python-side copy.
- Payload validation failure or unsupported descriptor always falls back to pickle unless explicitly forced by runtime config.

**Automatic stream transfer for large payloads:**

When the total buffer size exceeds a threshold (default 64 KB), Pulsing automatically uses a descriptor-first stream transfer instead of packing everything into a single message:

1. A lightweight descriptor header (dtype, shape, strides, buffer lengths) is sent as the first stream frame.
2. Buffer data follows as a sequence of raw chunk frames, each up to `PULSING_ZEROCOPY_CHUNK_BYTES` (default 1 MB).
3. The receiver pre-allocates buffers based on the descriptor and fills them incrementally as chunks arrive.

Small payloads below the threshold are still sent as a single message with descriptor + data packed together. This is transparent to the user — `actor.receive()` always gets a `ZeroCopyDescriptor` regardless of the transfer mode.

Environment variables:
- `PULSING_ZEROCOPY`: `auto` (default) / `off` / `force`
- `PULSING_ZEROCOPY_STREAM_THRESHOLD`: minimum buffer size in bytes to trigger stream transfer (default 65536)
- `PULSING_ZEROCOPY_CHUNK_BYTES`: chunk size in bytes for stream transfer (default 1048576, minimum 4096)

#### Actor Lifecycle

```python
from pulsing.actor import Actor, ActorId

class MyActor(Actor):
    def on_start(self, actor_id: ActorId):
        """Called when actor starts"""
        print(f"Started: {actor_id}")

    def on_stop(self):
        """Called when actor stops"""
        print("Stopping...")

    def metadata(self) -> dict[str, str]:
        """Return actor metadata (for diagnostics)"""
        return {"type": "worker", "version": "1.0"}

    async def receive(self, msg):
        return msg
```

#### Supervision and Restart Policies

```python
@pul.remote(
    restart_policy="on_failure",  # "never" | "on_failure" | "always"
    max_restarts=3,               # Maximum restart attempts
    min_backoff=0.1,              # Minimum backoff time (seconds)
    max_backoff=30.0,             # Maximum backoff time (seconds)
)
class ResilientWorker:
    def process(self, data):
        # Actor auto-restarts on exception
        return heavy_computation(data)
```

#### Streaming Responses

```python
@pul.remote
class StreamingService:
    # Return a generator, Pulsing auto-handles as streaming response
    async def generate_stream(self, n):
        for i in range(n):
            yield f"chunk_{i}"

    # Sync generators also supported
    def sync_stream(self, n):
        for i in range(n):
            yield f"item_{i}"

# Usage
service = await StreamingService.spawn()

# Client consumes stream
async for chunk in service.generate_stream(10):
    print(chunk)  # chunk_0, chunk_1, ...
```

**Note:** For `@pul.remote` classes, simply return a generator (sync or async) and Pulsing auto-detects and handles it as a streaming response.

### Queue API

Distributed queue with bucket-based partitioning, for data pipelines:

```python
import pulsing as pul

await pul.init()

# ── Write ──
writer = await pul.queue.write(
    "my_queue",
    *,
    bucket_column: str = "id",      # Column for partitioning
    num_buckets: int = 4,
    batch_size: int = 100,
    storage_path: str | None = None,
    backend: str = "memory",        # Pluggable: "memory" or custom
) -> QueueWriter

await writer.put({"id": "u1", "data": "hello"})
await writer.put([{"id": "u1", "data": "a"}, {"id": "u2", "data": "b"}])
await writer.flush()

# ── Read ──
reader = await pul.queue.read(
    "my_queue",
    *,
    bucket_id: int | None = None,
    bucket_ids: list[int] | None = None,
    rank: int | None = None,        # For distributed consumption
    world_size: int | None = None,
    num_buckets: int = 4,
) -> QueueReader

records = await reader.get(limit=100, wait=False)
```

### Topic API

Lightweight pub/sub for real-time message distribution:

```python
import pulsing as pul

await pul.init()

# ── Publish ──
writer = await pul.topic.write("events")
await writer.publish({"type": "user_login", "user": "alice"})

# ── Subscribe ──
reader = await pul.topic.read("events")

@reader.on_message
async def handle(msg):
    print(f"Received: {msg}")

await reader.start()
```

## Rust API

Rust API defines contracts via traits, organized in three layers:

### Quick Start

```rust
use pulsing_actor::prelude::*;

#[derive(Serialize, Deserialize)]
struct Ping(i32);

#[derive(Serialize, Deserialize)]
struct Pong(i32);

struct Echo;

#[async_trait]
impl Actor for Echo {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let Ping(x) = msg.unpack()?;
        Message::pack(&Pong(x))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    // Named actor (discoverable via resolve, uses namespace/name format)
    let actor = system.spawn_named("services/echo", Echo).await?;
    let Pong(x): Pong = actor.ask(Ping(1)).await?;

    // Anonymous actor (accessible only via ActorRef)
    let worker = system.spawn(Worker::new()).await?;

    system.shutdown().await?;
    Ok(())
}
```

### Trait Layers

#### ActorSystemCoreExt (Main path, auto-imported via prelude)

Core spawn and resolve capabilities:

```rust
// Spawn - Simple API
system.spawn(actor).await?;                    // Anonymous actor (not resolvable)
system.spawn_named(name, actor).await?;        // Named actor (resolvable)

// Spawn - Builder pattern (advanced configuration)
system.spawning()
    .name("services/counter")                  // Optional: named = resolvable
    .supervision(SupervisionSpec::on_failure().max_restarts(3))
    .mailbox_capacity(256)
    .spawn(actor).await?;

// Resolve - Simple
system.actor_ref(&actor_id).await?;            // By ActorId
system.resolve(name).await?;                   // By name

// Resolve - Builder pattern (advanced configuration)
system.resolving()
    .node(node_id)                             // Optional: target node
    .policy(RoundRobinPolicy::new())           // Optional: load balancing
    .filter_alive(true)                        // Optional: alive nodes only
    .resolve(name).await?;                     // Resolve single

system.resolving()
    .list(name).await?;                        // Get all instances

system.resolving()
    .lazy(name)?;                              // Lazy resolve
```

#### ActorSystemAdvancedExt (Advanced: restartable supervision)

Factory-pattern spawn with supervision restart (named actors only):

```rust
// Named actor + factory (restartable + resolvable)
// Note: anonymous actors don't support supervision (cannot re-resolve)
system.spawn_named_factory(name, || Ok(Service::new()), options).await?;
```

#### ActorSystemOpsExt (Operations / Diagnostics / Lifecycle)

System info, cluster membership, stop/shutdown:

```rust
system.node_id();
system.addr();
system.members().await;
system.all_named_actors().await;
system.stop(name).await?;
system.shutdown().await?;
```

### Key Conventions

- **Message encoding**: `Message::pack(&T)` uses bincode + `type_name::<T>()`; for cross-version protocols use `Message::single("TypeV1", bytes)`.
- **Optional zerocopy**: when payload objects implement `__zerocopy__(ctx)`, Pulsing may bypass pickle and send descriptor + buffers directly; otherwise it uses normal pickle/bytes paths.
- **Naming and resolution**:
  - `spawn_named(name, actor)`: Creates a discoverable actor, name is the resolution path
  - `resolve(name)`: One-shot resolve (may become stale after migration)
  - `resolve_lazy(name)`: Lazy resolve + auto-refresh (~5s TTL)
- **Streaming**: Return `Message::Stream`, cancellation is best-effort.
- **Supervision**: Only `spawn_named_factory` supports failure restart; anonymous actors do not support supervision.

### Behavior (Type-safe, Akka Typed style)

- **Core**: `Behavior<M>` + `TypedRef<M>` + `BehaviorAction (Same/Become/Stop)`
- **Constraint**: `TypedRef<M>` requires `M: Serialize + DeserializeOwned + Send + 'static`

Defined using function syntax, otherwise identical to Actor:

```rust
fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n, _ctx| {
        *count += n;
        BehaviorAction::Same
    })
}

// Behavior implements IntoActor, can be passed directly to spawn/spawn_named
// No manual wrapping needed, system converts automatically
let counter = system.spawn(counter(0)).await?;
let counter = system.spawn_named("actors/counter", counter(0)).await?;
```
