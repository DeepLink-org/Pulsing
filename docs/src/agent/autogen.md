# AutoGen Integration

Pulsing implements `PulsingRuntime`, a distributed runtime for Microsoft AutoGen.

## Features

- **Distributed Execution** - Run agents on different cluster nodes
- **Service Discovery** - Automatic agent discovery via Gossip protocol
- **Load Balancing** - Distribute workload across agent instances
- **API Compatible** - Drop-in replacement for `SingleThreadedAgentRuntime`

## Quick Start

```python
from pulsing.integrations.autogen import PulsingRuntime
from autogen_core import AgentId, RoutedAgent, message_handler

class MyAgent(RoutedAgent):
    @message_handler
    async def handle_msg(self, message: str, ctx) -> str:
        return f"Processed: {message}"

async def main():
    # Standalone mode
    runtime = PulsingRuntime()
    await runtime.start()

    # Register agent (same API as AutoGen)
    await runtime.register_factory("my_agent", lambda: MyAgent())

    # Send message
    response = await runtime.send_message("Hello", AgentId("my_agent", "default"))
    print(response)

    await runtime.stop()
```

## Distributed Mode

### Node 1 (Seed)

```python
runtime = PulsingRuntime(addr="0.0.0.0:8000")
await runtime.start()
await runtime.register_factory("writer", lambda: WriterAgent())
```

### Node 2 (Worker)

```python
runtime = PulsingRuntime(
    addr="0.0.0.0:8001",
    seeds=["192.168.1.100:8000"]
)
await runtime.start()
await runtime.register_factory("editor", lambda: EditorAgent())
```

### Any Node (Client)

```python
# Messages route automatically across cluster
await runtime.send_message(draft, AgentId("editor", "default"))
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `addr` | Node address | `None` (standalone) |
| `seeds` | Seed node addresses | `[]` |

## Example

```bash
cd examples/agent/autogen
./run_distributed.sh
```

## API Reference

### PulsingRuntime

```python
class PulsingRuntime:
    def __init__(self, addr: str | None = None, seeds: list[str] = [])
    async def start() -> None
    async def stop() -> None
    async def register_factory(type: str, factory: Callable) -> None
    async def send_message(message: Any, recipient: AgentId, sender: AgentId | None = None) -> Any
```
