# Remote Actors Guide

Guide to running and discovering actors across a cluster: setup, named actors, resolve.

## Before / After: Single Node vs Cluster

**Same Actor code.** Only initialization and how you get a reference change.

| | Single node (standalone) | Cluster (two nodes) |
|---|--------------------------|----------------------|
| **Init** | `await pul.init()` | Node 1: `await pul.init(addr="0.0.0.0:8000")`<br/>Node 2: `await pul.init(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])` |
| **Get actor** | `await Counter.spawn(value=0)` | Node 1: `await Counter.spawn(value=0, name="counter")`<br/>Node 2: `await Counter.resolve("counter")` |
| **Call** | `await counter.inc()` | Same: `await counter.inc()` — location transparent |

Once you have a proxy (from `spawn` or `resolve`), the API is identical. No “remote” vs “local” branches in your logic.

---

## Cluster Setup

### Starting a Seed Node

```python
import pulsing as pul

# Node 1: Start seed node
system = await pul.actor_system(addr="0.0.0.0:8000")

# Spawn a named actor (discoverable via resolve)
await system.spawn(WorkerActor(), name="worker")
```

### Joining a Cluster

```python
# Node 2: Join cluster
system = await pul.actor_system(
    addr="0.0.0.0:8001",
    seeds=["192.168.1.1:8000"]
)

# Wait for cluster sync
await asyncio.sleep(1.0)
```

## Finding Remote Actors

### Using pul.resolve() (recommended)

`pul.resolve()` returns an **ActorProxy** directly — no need to call `.as_type()` or `.as_any()`:

```python
# Typed proxy — when you know the class
proxy = await pul.resolve("worker", cls=Worker, timeout=30)
result = await proxy.process("hello")

# Untyped proxy — when remote type is unknown (any method)
proxy = await pul.resolve("worker", timeout=30)
result = await proxy.process("hello")
```

### Using @remote Class.resolve()

```python
@pul.remote
class Worker:
    def process(self, data): return f"processed: {data}"

# Same as pul.resolve("worker", cls=Worker)
worker = await Worker.resolve("worker")
result = await worker.process("hello")  # Direct method call
```

### Using system.resolve() (low-level)

When you need the raw **ActorRef** (e.g. for `.ask()` / `.tell()` or to pass to other APIs):

```python
remote_ref = await system.resolve("worker")
response = await remote_ref.ask({"action": "process", "data": "hello"})
# Get proxy from ref if needed: remote_ref.as_any() or remote_ref.as_type(Worker)
```

!!! note
    Prefer `pul.resolve(name, cls=...)` or `Class.resolve(name)` for a ready-to-use proxy. Use `system.resolve(name)` only when you need the low-level `ActorRef`.

## Named vs Anonymous Actors

### Named Actors (Discoverable)

Named actors are discoverable by any node in the cluster via `resolve()`:

```python
# Named actor - discoverable via resolve() from any node
await system.spawn(WorkerActor(), name="worker")

# Other nodes can find it by name
ref = await other_system.resolve("worker")
```

### Anonymous Actors (Local Reference Only)

Anonymous actors can only be accessed via the ActorRef returned by spawn:

```python
# Anonymous actor - only accessible via ActorRef
local_ref = await system.spawn(WorkerActor())

# Cannot be found via resolve(), only use the returned ActorRef
await local_ref.ask(msg)
```

## Location Transparency

Named actors support location transparency — same API for local and remote:

```python
# Local named actor
local_ref = await system.spawn(MyActor(), name="local-worker")

# Remote named actor (resolved via cluster)
remote_ref = await system.resolve("remote-worker")

# Exactly the same API for both
response1 = await local_ref.ask(msg)
response2 = await remote_ref.ask(msg)
```

## Error Handling

Pulsing provides unified error types for both local and remote actors, ensuring consistent error handling across the cluster.

### Error Types

- **PulsingRuntimeError**: Framework errors (network, cluster, actor system, etc.)
- **PulsingActorError**: Actor execution errors
  - **PulsingBusinessError**: Business logic errors (user input validation, etc.)
  - **PulsingSystemError**: System errors (may trigger actor restart)
  - **PulsingTimeoutError**: Timeout errors (retryable)

### Example

```python
from pulsing.exceptions import (
    PulsingBusinessError,
    PulsingSystemError,
    PulsingRuntimeError,
)

try:
    remote_ref = await system.resolve("worker")
    response = await remote_ref.ask(msg)
except PulsingBusinessError as e:
    # Handle business error (user input issue)
    print(f"Validation failed: {e.message}")
except PulsingSystemError as e:
    # Handle system error (may trigger restart)
    print(f"System error: {e.error}, recoverable: {e.recoverable}")
except PulsingRuntimeError as e:
    # Handle framework error (network, cluster, etc.)
    print(f"Framework error: {e}")
```

### Network Failures

Network-related errors are raised as `PulsingRuntimeError`:

```python
try:
    remote_ref = await system.resolve("worker")
    response = await remote_ref.ask(msg)
except PulsingRuntimeError as e:
    # Network failure, cluster issue, or actor not found
    if "Connection" in str(e) or "timeout" in str(e).lower():
        # Retry with backoff
        pass
    elif "not found" in str(e).lower():
        # Actor doesn't exist
        pass
```

### Timeouts

Use timeouts for remote calls to avoid indefinite waits:

```python
try:
    response = await asyncio.wait_for(remote_ref.ask(msg), timeout=10.0)
except asyncio.TimeoutError:
    print("Request timed out")
except PulsingRuntimeError as e:
    print(f"Remote call failed: {e}")
```

For proxy method calls: `await asyncio.wait_for(proxy.some_method(), timeout=10.0)`.

## Best Practices

1. **Wait for cluster sync**: Add a small delay after joining a cluster
2. **Handle errors gracefully**: Wrap remote calls in try-except blocks
3. **Use named actors**: Actors that need remote access must have a `name`
4. **Use pul.resolve(name, cls=...) or Class.resolve(name)**: Get typed proxies for better API experience
5. **Use timeouts**: Consider adding timeouts for remote calls

## Example: Distributed Counter

```python
import pulsing as pul

@pul.remote
class DistributedCounter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

# Node 1: Create named counter (discoverable remotely)
system1 = await pul.actor_system(addr="0.0.0.0:8000")
counter = await DistributedCounter.spawn(name="counter", init_value=0)

# Node 2: Access remote counter
system2 = await pul.actor_system(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])
await asyncio.sleep(1.0)

# Resolve and use the remote counter
remote_counter = await DistributedCounter.resolve("counter")
value = await remote_counter.get()  # 0
value = await remote_counter.increment(5)  # 5
```

## Next Steps

- Learn about [Actor System](actor_system.md) basics
- Check [Node Discovery](../design/node-discovery.md) for cluster details
