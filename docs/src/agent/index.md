# Agent Framework Integration

Pulsing provides native support for popular agent frameworks, enabling seamless scaling from single-process to distributed clusters.

## Design Philosophy

**Enhance, not replace.**

- Your existing agent code requires minimal changes
- APIs remain fully compatible with original frameworks
- One line of code to enable distributed execution

## Options

| Option | Integration | Description |
|--------|-------------|-------------|
| [**Pulsing Native**](native.md) | `@agent` | Lightweight toolkit with metadata support |
| [AutoGen](autogen.md) | `PulsingRuntime` | Drop-in replacement for default runtime |
| [LangGraph](langgraph.md) | `with_pulsing()` | Wrap compiled graphs |

## Pulsing Native Agent

For building multi-agent applications from scratch, use Pulsing's native `@agent` decorator:

```python
import pulsing as pul
from pulsing.agent import agent, llm, list_agents

@agent(role="Researcher", goal="Deep analysis")
class Researcher:
    async def analyze(self, topic: str) -> str:
        client = await llm()
        return await client.ainvoke(f"Analyze: {topic}")

await pul.init()
try:
    r = await Researcher.spawn(name="researcher")
    result = await r.analyze("AI trends")

    # Access metadata for visualization
    for name, meta in list_agents().items():
        print(f"{name}: {meta.role}")
finally:
    await pul.shutdown()
```

**Key features:**

- `@agent` = `@remote` + metadata (no magic)
- Metadata for visualization/debugging
- Full control over LLM calls

→ [Learn more about Pulsing Native Agent](native.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Agent Code                          │
│              (AutoGen / LangGraph / ...)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Pulsing Integration                        │
│          PulsingRuntime / with_pulsing()                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Pulsing Actor System                       │
│     Gossip Protocol │ Location Transparency │ Streaming     │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Node 1  │◄─►│  Node 2  │◄─►│  Node 3  │
    │   GPU    │   │   CPU    │   │   CPU    │
    └──────────┘   └──────────┘   └──────────┘
```

## Quick Comparison

### AutoGen

```python
from pulsing.integrations.autogen import PulsingRuntime

# Replace SingleThreadedAgentRuntime
runtime = PulsingRuntime(addr="0.0.0.0:8000")
await runtime.start()
await runtime.register_factory("agent", lambda: MyAgent())
```

### LangGraph

```python
from pulsing.integrations.langgraph import with_pulsing

app = graph.compile()
distributed_app = with_pulsing(
    app,
    node_mapping={"llm": "langgraph_node_llm"},
    seeds=["gpu-server:8001"],
)
```

## Use Cases

1. **LLM + Tool Separation** - LLM on GPU, tools on CPU
2. **Multi-Agent Collaboration** - Agents on different nodes
3. **Elastic Scaling** - Add/remove nodes without code changes

## Running Examples

```bash
# AutoGen
cd examples/agent/autogen && ./run_distributed.sh

# LangGraph
cd examples/agent/langgraph && ./run_distributed.sh
```
