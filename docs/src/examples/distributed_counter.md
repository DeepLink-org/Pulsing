# Distributed Counter (pattern)

This page documents a common pattern: a **named counter actor** that can be found from other nodes and updated remotely.

If you want a runnable baseline, start from `examples/python/named_actors.py` and `examples/python/cluster.py`, then apply the pattern below.

## Pattern

1. Start a seed node and spawn a **named actor** (discoverable via resolve)
2. Start worker nodes that join the cluster and **resolve the actor by name**
3. Use `ask` to update state and get a response

## Sketch

```python
import asyncio
import pulsing as pul


class Counter:
    def __init__(self):
        self.v = 0

    async def receive(self, msg):
        if msg.get("action") == "inc":
            self.v += msg.get("n", 1)
            return {"v": self.v}
        if msg.get("action") == "get":
            return {"v": self.v}
        return {}


async def seed():
    system = await pul.actor_system(addr="0.0.0.0:8000")
    # Named actors are automatically discoverable via resolve
    await system.spawn(Counter(), name="global_counter")
    await asyncio.Event().wait()


async def worker():
    system = await pul.actor_system(
        addr="0.0.0.0:8001",
        seeds=["127.0.0.1:8000"]
    )
    await asyncio.sleep(1.0)
    ref = await system.resolve("global_counter")
    resp = await ref.ask({"action": "inc", "n": 1})
    print(resp)


asyncio.run(asyncio.gather(seed(), worker()))
```
