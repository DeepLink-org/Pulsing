# Distributed Counter (pattern)

This page documents a common pattern: a **named counter actor** that can be found from other nodes and updated remotely.

If you want a runnable baseline, start from `examples/python/named_actors.py` and `examples/python/cluster.py`, then apply the pattern below.

## Pattern

1. Start a seed node and spawn a **public named actor**
2. Start worker nodes that join the cluster and **resolve the actor by name**
3. Use `ask` to update state and get a response

## Sketch

```python
import asyncio
from pulsing.actor import Actor, Message, SystemConfig, create_actor_system


class Counter(Actor):
    def __init__(self):
        self.v = 0

    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "Inc":
            self.v += int(msg.to_json().get("n", 1))
            return Message.from_json("Value", {"v": self.v})
        if msg.msg_type == "Get":
            return Message.from_json("Value", {"v": self.v})
        return Message.empty()


async def seed():
    system = await create_actor_system(SystemConfig.with_addr("0.0.0.0:8000"))
    await system.spawn("global_counter", Counter(), public=True)
    await asyncio.Event().wait()


async def worker():
    system = await create_actor_system(
        SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["127.0.0.1:8000"])
    )
    await asyncio.sleep(1.0)
    ref = await system.find("global_counter")
    resp = await ref.ask(Message.from_json("Inc", {"n": 1}))
    print(resp.to_json())


asyncio.run(asyncio.gather(seed(), worker()))
```
