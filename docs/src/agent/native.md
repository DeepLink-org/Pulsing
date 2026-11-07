# Actor Metadata Helpers

`pulsing.agent` retains a small set of actor helpers:

- `agent`: `@remote` with role/goal metadata for inspection.
- `runtime`: an async context manager for actor system initialization/shutdown.
- `get_agent_meta`, `list_agents`, and `cleanup`: metadata management.
- `parse_json` and `extract_field`: parsing utilities.

```python
import asyncio
from pulsing.agent import agent, cleanup, get_agent_meta, runtime


@agent(role="Worker", goal="Count messages")
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1
        return self.value


async def main():
    try:
        async with runtime():
            counter = await Counter.spawn(name="counter")
            print(await counter.inc())
            print(get_agent_meta("counter").role)
    finally:
        cleanup()


asyncio.run(main())
```

These helpers do not provide an LLM client, prompts, tool execution, or a coding
agent loop. Use application-owned libraries for those capabilities.

See [Remote Actors](../guide/remote_actors.md) for cluster setup and
`examples/agent/pulsing/runtime_lifecycle_example.py` for lifecycle examples.
