# Out-Cluster Connect

Access actors in a Pulsing cluster **without joining the cluster**.

`pulsing.connect` provides a lightweight connector that talks to any cluster node (acting as a gateway) and transparently routes requests to any actor. The connector does not participate in gossip, does not register as a node, and does not run fault detection — keeping it extremely lightweight.

## When to Use

| Scenario | Why Connect? |
|----------|-------------|
| **Notebook / CLI** | You only need to call actors, not host them |
| **Web backend** | Short-lived requests shouldn't trigger cluster join/leave churn |
| **Cross-network** | The caller may be in a different network zone |
| **Security isolation** | External callers should not see internal cluster topology |

!!! tip
    If your process also needs to **host** actors (be discoverable, respond to messages), use the regular `pul.init()` + cluster join instead. `Connect` is for **callers only**.

---

## Quick Start

```python
from pulsing.connect import Connect

# Connect to any cluster node (acts as gateway)
conn = await Connect.to("10.0.1.1:8080")

# Resolve a named actor
counter = await conn.resolve("services/counter")

# Call methods — same syntax as in-cluster ActorProxy
value = await counter.increment(5)
print(value)  # 5

# Close when done
await conn.close()
```

That's it. No `init()`, no cluster configuration, no ports to open.

---

## Communication Patterns

### Sync Methods

```python
conn = await Connect.to("10.0.1.1:8080")
calc = await conn.resolve("services/calculator")

result = await calc.multiply(6, 7)   # 42
await calc.add(10)                    # stateful call
value = await calc.get()              # 10
```

### Async Methods

Async methods on the remote actor work transparently:

```python
svc = await conn.resolve("services/ai")

# Async method — just await as usual
result = await svc.slow_process(data)
greeting = await svc.greet("world")  # "hello world"
```

### Streaming (Async Generators)

For actors that yield results incrementally (e.g., LLM token generation):

```python
llm = await conn.resolve("services/llm")

# Stream tokens as they're generated
async for token in llm.generate(prompt="Tell me a story"):
    print(token, end="", flush=True)
```

!!! note
    Streaming uses `async for proxy.method(args)` directly — **no `await`** before `async for`. The method call returns an async iterable.

### Remote Spawn

Spawn new actors on the cluster from the connector — the actor runs on a cluster node, not locally:

```python
from pulsing.connect import Connect
from myapp.actors import Calculator, Worker

conn = await Connect.to("10.0.1.1:8080")

# Spawn with constructor arguments
calc = await conn.spawn(Calculator, init_value=100, name="services/calc")
result = await calc.multiply(6, 7)  # 42

# Spawn an async actor
worker = await conn.spawn(Worker, name="services/worker")
status = await worker.process("task_data")
```

!!! note
    The `@remote`-decorated class must be registered on the target cluster node. This means the cluster process must have imported the class definition.

After spawning, the actor is fully accessible via `resolve()` too:

```python
# Another connector (or the same one) can resolve the spawned actor
calc = await conn.resolve("services/calc")
await calc.add(50)
```

### Concurrent Calls

Multiple calls can be made concurrently:

```python
import asyncio

svc = await conn.resolve("services/worker")
results = await asyncio.gather(
    svc.process("task_a"),
    svc.process("task_b"),
    svc.process("task_c"),
)
```

---

## Multiple Gateways (High Availability)

Pass a list of addresses for automatic failover:

```python
conn = await Connect.to([
    "10.0.1.1:8080",
    "10.0.1.2:8080",
    "10.0.1.3:8080",
])

# If the active gateway goes down, the connector fails over
# to the next available one.
```

Refresh the gateway list from the cluster at any time:

```python
await conn.refresh_gateways()
```

---

## Typed Resolve

If you have the actor class definition available, pass it to `resolve()` for method validation:

```python
from pulsing.connect import Connect
from myapp.actors import Calculator

conn = await Connect.to("10.0.1.1:8080")
calc = await conn.resolve("services/calc", cls=Calculator)

# Typo raises AttributeError immediately, not a remote error
calc.mulitply(6, 7)  # AttributeError: No method 'mulitply'
```

---

## Error Handling

### Actor Errors

Errors raised inside the remote actor propagate to the caller:

```python
from pulsing.exceptions import PulsingActorError

try:
    await calc.will_fail()
except PulsingActorError as e:
    print(f"Actor error: {e}")
```

### Streaming Errors

If an actor raises mid-stream, items received before the error are still available:

```python
items = []
try:
    async for item in svc.partial_stream(10):
        items.append(item)
except PulsingActorError as e:
    print(f"Stream error after {len(items)} items: {e}")
```

### Resolve Errors

Resolving a non-existent actor raises an error:

```python
try:
    await conn.resolve("services/nonexistent")
except Exception as e:
    print(f"Not found: {e}")
```

---

## In-Cluster vs Out-Cluster

| Capability | In-Cluster (`pul.init()`) | Out-Cluster (`Connect.to()`) |
|------------|:---:|:---:|
| `resolve` named actors | ✅ | ✅ |
| Sync method calls | ✅ | ✅ |
| Async method calls | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| `spawn` actors (remote) | ✅ | ✅ |
| `spawn` actors (local) | ✅ | ❌ |
| Be discoverable by others | ✅ | ❌ |
| Gossip membership | ✅ | ❌ |
| Open a port | ✅ | ❌ |

**Same actor code, same call syntax.** Only how you obtain the proxy differs.

---

## Best Practices

1. **Close connections** — Always call `await conn.close()` when done to release resources.
2. **Use multiple gateways** — Pass a list of addresses for production deployments.
3. **Handle errors** — Wrap remote calls in try-except blocks for robustness.
4. **Use typed resolve** — Pass `cls=` when you have the actor class for early error detection.
5. **Don't over-stream** — Use `await` for single results, `async for` only for actual streams.

---

## Next Steps

- [Remote Actors](remote_actors.md) — In-cluster actor communication
- [Communication Patterns](communication_patterns.md) — Sync, async, and streaming patterns
- [Design Doc: Out-Cluster Connect](../design/out-cluster-connect.md) — Architecture and protocol details
