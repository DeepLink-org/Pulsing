# Tutorial: Migrate from Ray

Migrate Ray actor code to Pulsing's native async API.

---

## Why This Migration Changed

This project no longer recommends the Ray-compatible layer (`pulsing.compat.ray`).
Use Pulsing's primary API directly:

- `import pulsing as pul`
- `@pul.remote`
- `await pul.init()` / `await pul.shutdown()`
- `await Class.spawn()` / `await Class.resolve()`

---

## API Mapping (Ray -> Pulsing)

| Ray | Pulsing |
|---|---|
| `ray.init()` | `await pul.init()` |
| `ray.shutdown()` | `await pul.shutdown()` |
| `@ray.remote` | `@pul.remote` |
| `Actor.remote(args...)` | `await Actor.spawn(args...)` |
| `ray.get(actor.method.remote(args...))` | `await actor.method(args...)` |
| `ray.get_actor(name)` | `await Actor.resolve(name)` or `await pul.resolve(name)` |

---

## Minimal Example

### Before (Ray)

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

### After (Pulsing)

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

- Prefer typed proxy: `await Class.resolve(name)`.
- If only a runtime name is available: `ref = await pul.resolve(name)` then `ref.as_type(Class)` / `ref.as_any()`.

---

## What's Next?

- [Guide: Actors](../guide/actors.md) — understand the Actor model
- [Guide: Remote Actors](../guide/remote_actors.md) — cluster setup
- [Tutorial: LLM Inference](llm_inference.md) — build an inference service
