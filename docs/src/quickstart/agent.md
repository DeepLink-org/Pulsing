# Tutorial: Distributed Agents

Integrate Pulsing with **AutoGen** and **LangGraph** to distribute your agents across machines.

**What you'll learn:**

- Use `PulsingRuntime` to distribute AutoGen agents
- Use `with_pulsing()` to distribute LangGraph nodes
- Run LLM on GPU, tools on CPU

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Agent Code                          │
│              (AutoGen / LangGraph)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Pulsing Actor System                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Node 1  │   │  Node 2  │   │  Node 3  │
    │   GPU    │   │   CPU    │   │   CPU    │
    │   LLM    │   │  Tools   │   │  Tools   │
    └──────────┘   └──────────┘   └──────────┘
```

---

## Prerequisites

```bash
pip install pulsing

# For AutoGen
pip install autogen-agentchat autogen-ext

# For LangGraph
pip install langgraph langchain-openai
```

---

## Option A: AutoGen

### Step 1: Define your agent

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model = OpenAIChatCompletionClient(model="gpt-4o-mini")
agent = AssistantAgent("assistant", model_client=model)
```

### Step 2: Use PulsingRuntime

Replace `SingleThreadedAgentRuntime` with `PulsingRuntime`:

```python
from pulsing.integrations.autogen import PulsingRuntime

# Single process (default)
runtime = PulsingRuntime()

# Distributed mode
runtime = PulsingRuntime(addr="0.0.0.0:8000")
```

### Step 3: Register and run

```python
await runtime.start()
await runtime.register_factory("assistant", lambda: agent)

# Send message
result = await runtime.send_message(
    "Hello, agent!",
    AgentId("assistant", "default")
)
```

### Full example

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_core import AgentId
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pulsing.integrations.autogen import PulsingRuntime

async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent("assistant", model_client=model)

    runtime = PulsingRuntime(addr="0.0.0.0:8000")
    await runtime.start()
    await runtime.register_factory("assistant", lambda: agent)

    result = await runtime.send_message(
        "What is 2 + 2?",
        AgentId("assistant", "default")
    )
    print(result)

asyncio.run(main())
```

---

## Option B: LangGraph

### Step 1: Build your graph

```python
from langgraph.graph import StateGraph

def llm_node(state):
    # LLM call
    return {"messages": [...]}

def tool_node(state):
    # Tool execution
    return {"messages": [...]}

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)
graph.add_edge("llm", "tool")
app = graph.compile()
```

### Step 2: Wrap with Pulsing

```python
from pulsing.integrations.langgraph import with_pulsing

distributed_app = with_pulsing(
    app,
    node_mapping={
        "llm": "langgraph_node_llm",    # → GPU server
        "tool": "langgraph_node_tool",  # → CPU server
    },
    seeds=["gpu-server:8001"],
)
```

### Step 3: Run

```python
result = await distributed_app.ainvoke({"messages": [...]})
```

---

## Running the Examples

```bash
# AutoGen distributed
cd examples/agent/autogen && ./run_distributed.sh

# LangGraph distributed
cd examples/agent/langgraph && ./run_distributed.sh
```

---

## Use Cases

| Scenario | Setup |
|----------|-------|
| **LLM + Tools** | LLM on GPU node, tools on CPU nodes |
| **Multi-Agent** | Each agent on different node |
| **Elastic Scaling** | Add/remove nodes without code changes |

---

## What's Next?

- [Agent: AutoGen](../agent/autogen.md) — detailed AutoGen integration
- [Agent: LangGraph](../agent/langgraph.md) — detailed LangGraph integration
- [Guide: Remote Actors](../guide/remote_actors.md) — cluster setup
