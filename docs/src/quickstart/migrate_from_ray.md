# Tutorial: Migrate from Ray

Replace Ray with Pulsing in **5 minutes**. One import change, zero external dependencies.

---

## Why Migrate?

| | Ray | Pulsing |
|---|-----|---------|
| **Dependencies** | Ray cluster, Redis, GCS | None |
| **Startup time** | Seconds | Milliseconds |
| **Memory overhead** | High | Low |
| **Actor model** | Stateful remote objects | Classical (mailbox, FIFO) |
| **Streaming** | Manual | Native |

---

## Step 1: Change the Import

```python
# Before (Ray)
import ray

# After (Pulsing)
from pulsing.compat import ray
```

**That's it.** Your existing code works.

---

## Step 2: Run Your Code

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
print(ray.get(counter.inc.remote()))  # 1
print(ray.get(counter.inc.remote()))  # 2

ray.shutdown()
```

### After (Pulsing)

```python
from pulsing.compat import ray  # ← only this line changed

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1
        return self.value

counter = Counter.remote()
print(ray.get(counter.inc.remote()))  # 1
print(ray.get(counter.inc.remote()))  # 2

ray.shutdown()
```

---

## Supported APIs

| API | Status |
|-----|--------|
| `ray.init()` | ✅ |
| `ray.shutdown()` | ✅ |
| `@ray.remote` (class) | ✅ |
| `@ray.remote` (function) | ✅ |
| `ray.get()` | ✅ |
| `ray.put()` | ✅ |
| `ray.wait()` | ✅ |
| `ActorClass.remote()` | ✅ |
| `actor.method.remote()` | ✅ |

---

## Distributed Mode

Ray requires a cluster. Pulsing just needs `--addr` and `--seeds`:

### Node 1 (seed)

```python
from pulsing.compat import ray

ray.init(address="0.0.0.0:8000")

@ray.remote
class Worker:
    def process(self, data):
        return f"processed: {data}"

worker = Worker.remote()
# Keep running...
```

### Node 2 (join)

```python
from pulsing.compat import ray

ray.init(address="0.0.0.0:8001", seeds=["192.168.1.1:8000"])

# Find remote actor
worker = ray.get_actor("Worker")
result = ray.get(worker.process.remote("hello"))
```

---

## Native Async API (Optional)

For new code, consider the native async API:

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
    counter = await Counter.spawn()
    print(await counter.inc())  # 1
    await pul.shutdown()
```

**Benefits:**

- Cleaner `async/await` syntax
- No `ray.get()` boilerplate
- IDE autocompletion works
- Access to streaming messages

---

## Limitations

The Ray-compatible API does not support:

- Ray Serve
- Ray Tune
- Ray Data
- Object Store (large objects)
- Placement Groups

For these features, continue using Ray. Pulsing focuses on the Actor model.

---

## What's Next?

- [Guide: Actors](../guide/actors.md) — understand the Actor model
- [Guide: Remote Actors](../guide/remote_actors.md) — cluster setup
- [Tutorial: LLM Inference](llm_inference.md) — build an inference service
