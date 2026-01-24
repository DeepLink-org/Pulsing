# API Reference

Complete API documentation for Pulsing Actor Framework.

## Contract & Semantics (Derived from `llms.binding.md`)

This section is the **user-facing contract** for Pulsing's Python API. It is **derived from** the repository document **`llms.binding.md`**.

- **Source of truth**: `llms.binding.md` is the canonical contract.
- **This page**: API reference + explicit semantics (concurrency, errors, trust boundaries).
- **If there's a mismatch**: treat `llms.binding.md` as authoritative; please open an issue/PR to sync docs.

### Concurrency model for `@pulsing.remote`

For a `@pulsing.remote` class, method calls are translated into actor messages.

- **Sync method (`def method`)**
  - Executed **serially** (one request at a time) in the actor.
  - Recommended for fast CPU work and state mutation.
- **Async method (`async def method`)**
  - The call uses **stream-backed execution** and is scheduled as a background task on the actor side.
  - While the method is awaiting, the actor can continue receiving other messages (**non-blocking** behavior).
  - You can either:
    - `await proxy.async_method(...)` to get the final value, or
    - `async for chunk in proxy.async_method(...): ...` to consume streamed yields.
- **Generators (sync/async)**
  - Returning a generator (sync or async) is treated as a **streaming response**.

### Streaming & cancellation

- Streaming is implemented via Pulsing stream messages; cancellation is **best-effort**.
- If a caller cancels the local await/iteration, the remote side may or may not stop immediately, depending on transport-level cancellation propagation.

### `ask` vs `tell`

- **`ask(msg)`**: request/response. Returns a value (or raises).
- **`tell(msg)`**: fire-and-forget. No response is awaited.

### Error model (current behavior)

- Actor-side exceptions are transported back and typically raised as **`RuntimeError(str(e))`** on the caller side.
- Timeout helpers (where used) raise **`asyncio.TimeoutError`**.

Note: error *type information and remote stack traces* are not guaranteed to be preserved.

### Trust boundary & security notes

- **Pickle-based payloads (Python ↔ Python)**:
  - Python-to-Python payloads are transported as **pickle** by default for convenience.
  - **Risk**: unpickling untrusted data can lead to arbitrary code execution (RCE).
  - **Guideline**: only use pickle payloads inside a **trusted network / trusted cluster** boundary.
- **Transport security (TLS)**:
  - For production deployments, always enable TLS and treat the cluster as an authenticated trust boundary.

### Queue semantics (distributed queue)

- **Bucketing**:
  - Writer uses `bucket_column` + `num_buckets` to partition records into buckets.
  - Readers must use a consistent `num_buckets` (and backend) with writers.
- **Ownership**:
  - Bucket ownership is computed by hashing over live cluster members; requests may be redirected to the owning node.
- **Backends**:
  - Default backend is in-memory; persistence depends on the selected backend.

## Core Functions

### pul.actor_system

Create a new Actor System instance.

```python
import pulsing as pul

system = await pul.actor_system(
    addr: str | None = None,        # Bind address, None for standalone
    *,
    seeds: list[str] | None = None, # Seed nodes for cluster
    passphrase: str | None = None,  # TLS passphrase
) -> ActorSystem
```

**Example:**

```python
# Standalone mode
system = await pul.actor_system()

# Cluster mode
system = await pul.actor_system(addr="0.0.0.0:8000")

# Join existing cluster
system = await pul.actor_system(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])

# Shutdown
await system.shutdown()
```

### pul.init / pul.shutdown

Global system initialization (Ray-style async API).

```python
import pulsing as pul

# Initialize global system
await pul.init(addr=None, seeds=None, passphrase=None)

# Use global system
actor = await pul.spawn(MyActor())
ref = await pul.resolve("actor_name")

# Shutdown
await pul.shutdown()
```

## Core Classes

### ActorSystem

Main entry point for the actor system.

```python
class ActorSystem:
    async def spawn(
        self,
        actor: Actor,
        *,
        name: str | None = None,
        # public parameter is deprecated: all named actors are resolvable
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0
    ) -> ActorRef:
        """
        Spawn a new actor.

        - With name: named actor, discoverable via resolve()
        - Without name: anonymous actor, only accessible via returned ActorRef
        """
        pass

    async def refer(self, actorid: ActorId | str) -> ActorRef:
        """Get ActorRef by ActorId."""
        pass

    async def resolve(self, name: str, *, node_id: int | None = None) -> ActorRef:
        """Resolve actor by name."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the actor system."""
        pass
```

### ActorRef

Low-level reference to an actor. Use `ask()` and `tell()` to communicate.

```python
class ActorRef:
    @property
    def actor_id(self) -> ActorId:
        """Get the actor's ID."""
        pass

    async def ask(self, msg: Any) -> Any:
        """Send a message and wait for response."""
        pass

    async def tell(self, msg: Any) -> None:
        """Send a message without waiting for response (fire-and-forget)."""
        pass
```

### ActorProxy

High-level proxy for `@remote` classes. Call methods directly.

```python
class ActorProxy:
    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef."""
        pass

    # Call methods directly:
    # result = await proxy.my_method(arg1, arg2)
```

## Decorators

### @remote / @pul.remote

Convert a class into a distributed Actor.

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    # Sync method - sequential execution
    def incr(self) -> int:
        self.value += 1
        return self.value

    # Async method - concurrent execution during await
    async def fetch_and_add(self, url: str) -> int:
        data = await http_get(url)
        self.value += data
        return self.value

    # Generator - automatic streaming
    async def stream(self):
        for i in range(10):
            yield {"count": i}

# Create actor
counter = await Counter.spawn(name="counter")

# Call methods directly
result = await counter.incr()

# Streaming
async for chunk in counter.stream():
    print(chunk)

# Resolve existing actor
proxy = await Counter.resolve("counter")
```

**Supervision parameters:**

```python
@pul.remote(
    restart_policy="on_failure",  # "never" | "on_failure" | "always"
    max_restarts=3,
    min_backoff=0.1,
    max_backoff=30.0,
)
class ResilientWorker:
    def work(self, data): ...
```

## Base Actor

For low-level control, inherit from Actor base class.

```python
class MyActor:
    def __init__(self):
        self.value = 0

    def on_start(self, actor_id):
        """Called when actor starts."""
        print(f"Started: {actor_id}")

    async def receive(self, msg):
        """Handle incoming messages."""
        if msg.get("action") == "add":
            self.value += msg.get("n", 1)
            return {"value": self.value}
        return {"error": "unknown action"}

# Spawn
system = await pul.actor_system()
actor = await system.spawn(MyActor(), name="my_actor")

# Communicate via ask/tell
response = await actor.ask({"action": "add", "n": 10})
```

## Queue API

Distributed queue for data pipelines.

```python
# Write
writer = await system.queue.write(
    topic="my_queue",
    bucket_column="user_id",
    num_buckets=4,
)
await writer.put({"user_id": "u1", "data": "hello"})
await writer.flush()

# Read
reader = await system.queue.read("my_queue")
records = await reader.get(limit=100)
```

## Ray Compatibility

Drop-in replacement for Ray.

```python
from pulsing.compat import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    def incr(self):
        self.value += 1
        return self.value

counter = Counter.remote()
result = ray.get(counter.incr.remote())

ray.shutdown()
```

## Rust API

The Rust API is organized into three trait layers (all re-exported in `pulsing_actor::prelude::*`):

### ActorSystemCoreExt (Primary API)

Core spawn and resolve operations:

```rust
// Spawn - Simple API
system.spawn(actor).await?;                    // Anonymous actor (not resolvable)
system.spawn_named(name, actor).await?;        // Named actor (resolvable)

// Spawn - Builder pattern (advanced config)
system.spawning()
    .name("services/counter")                  // Optional: with name = resolvable
    .supervision(SupervisionSpec::on_failure().max_restarts(3))
    .mailbox_capacity(256)
    .spawn(actor).await?;

// Resolve - Simple API
system.actor_ref(&actor_id).await?;            // Get by ActorId
system.resolve(name).await?;                   // Resolve by name

// Resolve - Builder pattern (advanced config)
system.resolving()
    .node(node_id)                             // Optional: target node
    .policy(RoundRobinPolicy::new())           // Optional: load balancing
    .filter_alive(true)                        // Optional: only alive nodes
    .resolve(name).await?;                     // Resolve single

system.resolving().list(name).await?;          // Get all instances
system.resolving().lazy(name)?;                // Lazy resolution (~5s TTL auto-refresh)
```

### ActorSystemAdvancedExt (Supervision/Restart)

Factory-based spawning for supervision restarts (named actors only):

```rust
let options = SpawnOptions::new()
    .supervision(SupervisionSpec::on_failure().max_restarts(3));

// Only named actors support supervision (anonymous cannot be re-resolved)
system.spawn_named_factory(name, || Ok(Service::new()), options).await?;
```

### ActorSystemOpsExt (Operations/Diagnostics)

System info, cluster membership, lifecycle:

```rust
system.node_id();
system.addr();
system.members().await;
system.all_named_actors().await;
system.stop(name).await?;
system.shutdown().await?;
```

## Examples

See the [Quick Start Guide](quickstart/index.md) for usage examples.
