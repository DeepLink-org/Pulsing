# Ping-Pong

The simplest actor communication example.

## Code

```python
import asyncio
import pulsing as pul


class PingPong:
    async def receive(self, msg):
        if msg == "ping":
            return "pong"
        return f"echo: {msg}"


async def main():
    system = await pul.actor_system()
    actor = await system.spawn(PingPong())

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

- Implement `receive()` to handle messages
- **Any Python object** can be a message (string, dict, list, etc.)
- `actor.ask(msg)` sends a message and waits for response
- `system.shutdown()` cleanly stops all actors
