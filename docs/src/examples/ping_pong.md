# Ping-Pong

The simplest actor communication example.

## Code

```python
import asyncio
from pulsing.actor import Actor, SystemConfig, create_actor_system


class PingPong(Actor):
    async def receive(self, msg):
        if msg == "ping":
            return "pong"
        return f"echo: {msg}"


async def main():
    system = await create_actor_system(SystemConfig.standalone())
    actor = await system.spawn("pingpong", PingPong())

    print(await actor.ask("ping"))   # -> pong
    print(await actor.ask("hello"))  # -> echo: hello

    await system.shutdown()


asyncio.run(main())
```

## Run

```bash
python examples/python/ping_pong.py
```

## Key Points

- `Actor` is the base class - implement `receive()` to handle messages
- **Any Python object** can be a message (string, dict, list, etc.)
- `actor.ask(msg)` sends a message and waits for response
- `system.shutdown()` cleanly stops all actors
