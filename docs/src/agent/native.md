# Pulsing Native Agent

Pulsing provides a lightweight native agent toolkit for building multi-agent applications, fully compatible with the Actor model.

## Design Philosophy

**Simple and transparent.**

- `@agent` = `@remote` + metadata (no magic)
- Metadata for visualization/debugging, not auto-prompt injection
- Full control over LLM calls and conversation flow

## Core API

### `@agent` Decorator

The `@agent` decorator is equivalent to `@remote`, but attaches metadata for visualization and debugging:

```python
from pulsing.actor import remote, resolve
from pulsing.agent import agent, runtime, llm, get_agent_meta, list_agents

# @remote: Basic Actor
@remote
class Worker:
    async def work(self):
        return "done"

# @agent: Actor with metadata
@agent(role="Researcher", goal="Deep analysis", domain="AI")
class Researcher:
    async def analyze(self, topic: str) -> str:
        client = await llm()
        resp = await client.ainvoke(f"Analyze: {topic}")
        return resp.content
```

### Metadata Access

```python
async with runtime():
    r = await Researcher.spawn(name="researcher")

    # Get metadata by name
    meta = get_agent_meta("researcher")
    print(meta.role)       # "Researcher"
    print(meta.goal)       # "Deep analysis"
    print(meta.tags)       # {"domain": "AI"}

    # List all agents
    for name, meta in list_agents().items():
        print(f"{name}: {meta.role}")
```

### `@remote` vs `@agent`

| Feature | `@remote` | `@agent` |
|---------|-----------|----------|
| Function | Actor wrapper | Actor wrapper + metadata |
| Use case | General purpose | Visualization / debugging |
| Metadata | None | `role`, `goal`, `backstory`, `tags` |
| Performance | Baseline | Nearly identical |

## Runtime Management

```python
from pulsing.agent import runtime, cleanup

async with runtime():
    # Create and use agents
    agent = await MyAgent.spawn(name="agent")
    await agent.work()

# Optional: cleanup global state
cleanup()
```

### Distributed Mode

```python
# Node A
async with runtime(addr="0.0.0.0:8001"):
    await JudgeActor.spawn(name="judge")

# Node B (auto-discovers Node A)
async with runtime(addr="0.0.0.0:8002", seeds=["node_a:8001"]):
    judge = await resolve("judge")  # Cross-node transparent call
    await judge.submit(idea)
```

## LLM Integration

```python
from pulsing.agent import llm

async def analyze(topic: str):
    # Get LLM client (lazy-loaded singleton)
    client = await llm(temperature=0.8)
    resp = await client.ainvoke(f"Analyze: {topic}")
    return resp.content
```

**Environment Variables:**

- `OPENAI_API_KEY`: API key (required)
- `OPENAI_BASE_URL`: Custom API endpoint (optional)
- `LLM_MODEL`: Default model name (optional, defaults to `gpt-4o-mini`)

## Utility Functions

### JSON Parsing

```python
from pulsing.agent import parse_json, extract_field

# Safe JSON parsing with fallback
data = parse_json('{"key": "value"}', default={})

# Extract specific field
value = extract_field(response, "answer", default="unknown")
```

## Complete Example

```python
import asyncio
from pulsing.actor import remote, resolve
from pulsing.agent import agent, runtime, llm, parse_json, get_agent_meta

@remote
class Moderator:
    """Coordinator using @remote (basic Actor)"""

    def __init__(self, topic: str):
        self.topic = topic
        self.opinions = []

    async def collect_opinion(self, agent_name: str, opinion: str):
        self.opinions.append({"agent": agent_name, "opinion": opinion})
        return {"received": True}

    async def summarize(self):
        return {"topic": self.topic, "opinions": self.opinions}

@agent(role="Analyst", goal="Provide insights", domain="tech")
class Analyst:
    """Analyst using @agent (Actor with metadata)"""

    def __init__(self, name: str, moderator: str, mock: bool = True):
        self.name = name
        self.moderator_name = moderator
        self.mock = mock

    async def analyze(self, topic: str):
        if self.mock:
            opinion = f"[{self.name}] Analysis of {topic}: looks promising"
        else:
            client = await llm()
            resp = await client.ainvoke(f"Brief analysis of: {topic}")
            opinion = resp.content

        # Submit to moderator
        moderator = await resolve(self.moderator_name)
        await moderator.collect_opinion(self.name, opinion)
        return opinion

async def main():
    async with runtime():
        # Create moderator
        moderator = await Moderator.spawn(topic="AI Trends", name="moderator")

        # Create analysts
        for i in range(3):
            name = f"analyst_{i}"
            await Analyst.spawn(
                name=name,
                moderator="moderator",
                mock=True,
                name=name,
            )

        # Show agent metadata
        print("Registered agents:")
        for name, meta in list_agents().items():
            print(f"  {name}: {meta.role}")

        # Run analysis
        for i in range(3):
            analyst = await resolve(f"analyst_{i}")
            await analyst.analyze("AI Trends")

        # Get summary
        result = await moderator.summarize()
        print(f"Summary: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Examples

See `examples/agent/pulsing/` for complete examples:

| Example | Description |
|---------|-------------|
| `mbti_discussion.py` | Multi-agent discussion with MBTI personalities |
| `parallel_ideas_async.py` | Parallel idea generation with competitive submission |
| `runtime_lifecycle_example.py` | Runtime lifecycle management |

```bash
# Run MBTI discussion (mock mode)
python examples/agent/pulsing/mbti_discussion.py --mock --group-size 6

# Run parallel ideas (mock mode)
python examples/agent/pulsing/parallel_ideas_async.py --mock --n-ideas 5
```

## What's Next?

- [AutoGen Integration](autogen.md) — Use Pulsing with AutoGen
- [LangGraph Integration](langgraph.md) — Use Pulsing with LangGraph
- [Remote Actors](../guide/remote_actors.md) — Cluster setup
