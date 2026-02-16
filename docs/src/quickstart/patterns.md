# Actor Patterns

Common patterns right after your first Actor: named actors, resolve, and when to use ask vs tell.

---

## Named actors and resolve

Give an actor a **name** so other code can find it with **resolve** (same process or across the cluster):

```python
import pulsing as pul

@pul.remote
class Worker:
    def process(self, data: str) -> str:
        return f"processed: {data}"

async def main():
    await pul.init()
    # Spawn with a name — discoverable via resolve
    await Worker.spawn(name="worker")
    # Later (or on another node): get a proxy by name
    worker = await Worker.resolve("worker")
    result = await worker.process("hello")
    await pul.shutdown()
```

Anonymous actors (no `name=`) are only reachable via the `ActorRef` returned by `spawn()`.

---

## Ask vs tell

| Pattern | Method | Use when |
|--------|--------|----------|
| **Request–response** | `await ref.ask(msg)` or `await proxy.method()` | You need a return value. |
| **Fire-and-forget** | `await ref.tell(msg)` | You don't need a reply; best-effort delivery. |

For typed proxies, method calls are like **ask** (they return the result). Use **tell** when you have an `ActorRef` and want to send without waiting.

---

## Next steps

- [Cluster Setup](cluster_networking.md) — form a cluster (Gossip / Head / Ray)
- [Actor Basics](../guide/actors.md) — deeper model and API
- [Communication Patterns](../guide/communication_patterns.md) — streaming, timeouts, and more
