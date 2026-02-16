# LangGraph Integration

Pulsing provides `with_pulsing()`, a one-line wrapper for distributed LangGraph execution.

## Features

- **One-line Integration** - Wrap any compiled graph
- **Node Mapping** - Route specific nodes to remote workers
- **API Compatible** - Same `ainvoke()`, `invoke()`, `astream()` methods
- **State Serialization** - Automatic state transfer between nodes

## Quick Start

```python
from pulsing.integrations.langgraph import with_pulsing
from langgraph.graph import StateGraph

# Build graph as usual
graph = StateGraph(MyState)
graph.add_node("llm", llm_fn)
graph.add_node("tool", tool_fn)
app = graph.compile()

# One line to enable distributed execution
distributed_app = with_pulsing(
    app,
    node_mapping={
        "llm": "langgraph_node_llm",
        "tool": "langgraph_node_tool",
    },
    seeds=["gpu-server:8001"],
)

# Use exactly the same way
result = await distributed_app.ainvoke({"messages": [...]})
```

## Node Mapping

The `node_mapping` parameter defines which nodes execute remotely:

- **Key**: LangGraph node name
- **Value**: Pulsing Actor name on remote worker
- **Unmapped nodes**: Execute locally

```python
node_mapping={
    "llm": "langgraph_node_llm",    # LLM → GPU server
    "tool": "langgraph_node_tool",  # Tool → CPU server
}
```

## Starting Workers

```python
from pulsing.integrations.langgraph import start_worker

# GPU server
await start_worker("llm", llm_node, addr="0.0.0.0:8001")

# CPU server (joins cluster)
await start_worker("tool", tool_node, addr="0.0.0.0:8002", seeds=["gpu:8001"])
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `node_mapping` | Node to Actor mapping | `{}` (all local) |
| `addr` | This node's address | `None` (auto) |
| `seeds` | Seed node addresses | `[]` |

## Example

```bash
cd examples/agent/langgraph
./run_distributed.sh
```

## API Reference

### with_pulsing

```python
def with_pulsing(
    compiled_graph,
    *,
    node_mapping: dict[str, str] | None = None,
    addr: str | None = None,
    seeds: list[str] | None = None,
) -> PulsingGraphWrapper
```

### start_worker

```python
async def start_worker(
    node_name: str,
    node_func: Callable,
    *,
    addr: str,
    seeds: list[str] | None = None,
    actor_name: str | None = None,
) -> None
```

### PulsingGraphWrapper

```python
class PulsingGraphWrapper:
    async def ainvoke(input: dict, config: dict | None = None, **kwargs) -> dict
    def invoke(input: dict, config: dict | None = None, **kwargs) -> dict
    async def astream(input: dict, config: dict | None = None, **kwargs) -> AsyncIterator[dict]
    async def close() -> None
```
