# Tutorial: Ray + Pulsing

Use Pulsing as the communication backbone for your Ray actors — add streaming, actor discovery, and cross-cluster calls without replacing Ray.

---

## Two Ways to Use Pulsing with Ray

1. **Bridge mode** — Keep your Ray actors, add Pulsing communication via `pul.mount()`
2. **Standalone mode** — Use Pulsing's native API directly (for new projects or full migration)

---

## Bridge Mode: Add Pulsing to Ray Actors

The simplest path — keep Ray for scheduling, add Pulsing for communication:

```python
import ray
import pulsing as pul

@ray.remote
class Worker:
    def __init__(self, name):
        pul.mount(self, name=name)  # One line: join the Pulsing network

    async def call_peer(self, peer_name, msg):
        proxy = await pul.resolve(peer_name, timeout=30)
        return await proxy.greet(msg)  # Cross-process Pulsing call

    async def greet(self, msg):
        return f"hello: {msg}"

ray.init()
workers = [Worker.remote(f"w{i}") for i in range(3)]
ray.get(workers[0].call_peer.remote("w1", "hi"))  # => "hello: hi"
pul.cleanup_ray()
```

**What you get:** Ray handles process scheduling and resource management. Pulsing adds streaming, named actor discovery, and direct actor-to-actor communication — without going through Ray's object store.

---

## Standalone Mode: Pulsing Native API

For new projects or when you want Pulsing's full feature set:

### API Mapping (Ray -> Pulsing)

| Ray | Pulsing |
|---|---|
| `ray.init()` | `await pul.init()` |
| `ray.shutdown()` | `await pul.shutdown()` |
| `@ray.remote` | `@pul.remote` |
| `Actor.remote(args...)` | `await Actor.spawn(args...)` |
| `ray.get(actor.method.remote(args...))` | `await actor.method(args...)` |
| `ray.get_actor(name)` | `await Actor.resolve(name)` or `await pul.resolve(name)` |

### Minimal Example

**Ray:**

```python
import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    def inc(self):
        self.value += 1
        return self.value

counter = Counter.remote()
print(ray.get(counter.inc.remote()))
ray.shutdown()
```

**Pulsing:**

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self):
        self.value = 0
    def inc(self):
        self.value += 1
        return self.value

async def main():
    await pul.init()
    counter = await Counter.spawn(name="counter")
    print(await counter.inc())
    await pul.shutdown()
```

**Key differences:**

| Aspect | Ray | Pulsing |
|--------|-----|---------|
| Create actor | `Counter.remote()` | `await Counter.spawn()` — native async |
| Call method | `ray.get(counter.inc.remote())` | `await counter.inc()` — direct await |
| Get by name | `ray.get_actor("counter")` | `await Counter.resolve("counter")` — typed proxy |
| Streaming | Not built-in | Native `async for chunk in actor.stream()` |
| Discovery | Needs GCS | Built-in gossip, zero external deps |

Same mental model (remote class, spawn, method calls). Pulsing adds native async, streaming, and self-contained clustering.

---

## Distributed Mode Mapping

### Node 1 (seed)

```python
import pulsing as pul

@pul.remote
class Worker:
    def process(self, data: str) -> str:
        return f"processed: {data}"

await pul.init(addr="0.0.0.0:8000")
await Worker.spawn(name="worker")
```

### Node 2 (join + resolve)

```python
import pulsing as pul

await pul.init(addr="0.0.0.0:8001", seeds=["192.168.1.1:8000"])
worker = await Worker.resolve("worker")
result = await worker.process("hello")
```

---

## Notes

- Prefer typed proxy: `await pul.resolve(name, cls=Class)` or `await Class.resolve(name)`.
- If only a runtime name is available: `proxy = await pul.resolve(name)` — you can call any method directly (untyped).

---

## What's Next?

- [Guide: Actors](../guide/actors.md) — understand the Actor model
- [Guide: Remote Actors](../guide/remote_actors.md) — cluster setup
- [Tutorial: LLM Inference](llm_inference.md) — build an inference service
