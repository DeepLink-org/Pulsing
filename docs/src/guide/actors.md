# Actor Guide

This guide covers the **Actor model** concepts and patterns for building robust distributed applications.

!!! tip "Prerequisite"
    If you haven't completed the [Quickstart](../quickstart/index.md), start there first.

!!! tip "Communication Patterns"
    Not sure when to use sync vs async vs streaming? See the [Communication Patterns Guide](communication_patterns.md) for detailed guidance.

---

## What is an Actor?

An **Actor** is a fundamental unit of computation in concurrent and distributed systems. The Actor model, introduced by Carl Hewitt in 1973, provides a principled approach to building systems that are:

- **Concurrent**: Multiple actors run in parallel
- **Distributed**: Actors can be on different machines
- **Fault-tolerant**: Failures are isolated

### Core Principles

```
┌─────────────────────────────────────────────────────────────┐
│                         Actor                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Private     │    │  Mailbox    │    │  Behavior   │     │
│  │ State       │    │  (FIFO)     │    │  (Methods)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        ▲                  │                   │             │
│        │                  ▼                   ▼             │
│        └──────── process one message at a time ────────────│
└─────────────────────────────────────────────────────────────┘
```

| Principle | Description |
|-----------|-------------|
| **Isolation** | Each actor has private state; no shared memory |
| **Message-passing** | Actors communicate only via async messages |
| **Sequential processing** | One message at a time (no internal locks) |
| **Location transparency** | Same API for local and remote actors |

### Why Pulsing over Ray?

Ray's "actor" is essentially a **stateful remote object** — you call methods on it, but there's no formal message queue or delivery semantics.

Pulsing follows the **classical Actor model** (like Erlang/Akka):

| Feature | Pulsing | Ray |
|---------|---------|-----|
| Message queue (mailbox) | ✅ FIFO | ❌ Direct call |
| Sequential guarantee | ✅ Per-actor | ⚠️ Per-method |
| Supervision/restart | ✅ Built-in | ❌ Manual |
| Zero external deps | ✅ | ❌ (needs Ray cluster) |
| Streaming messages | ✅ Native | ❌ |

---

## Two API Styles

| API | Import | Style | Best For |
|-----|--------|-------|----------|
| **Native Async** | `import pulsing as pul` | `async/await` | New projects, maximum performance |
| **Ray-Compatible** | `from pulsing.compat import ray` | Synchronous | Migrating from Ray, quick prototyping |

### Native Async API (Recommended)

```python
import pulsing as pul

@pul.remote
class Calculator:
    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def add(self, n: int) -> int:
        self.value += n
        return self.value

async def main():
    await pul.init()
    calc = await Calculator.spawn(initial_value=100)
    result = await calc.add(50)  # 150
    await pul.shutdown()
```

### Ray-Compatible API

```python
from pulsing.compat import ray

ray.init()

@ray.remote
class Calculator:
    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def add(self, n: int) -> int:
        self.value += n
        return self.value

calc = Calculator.remote(initial_value=100)
result = ray.get(calc.add.remote(50))  # 150
ray.shutdown()
```

**Migration from Ray** — just change the import:

```python
# Before:  import ray
# After:   from pulsing.compat import ray
```

---

## Message Patterns

### Ask (Request-Response)

```python
result = await calc.add(10)
```

### Tell (Fire-and-forget)

```python
await actor_ref.tell({"event": "notify", "data": "event_data"})
```

### Streaming Messages

For continuous data flow (e.g., LLM token generation), just return a generator:

```python
@pul.remote
class TokenGenerator:
    async def generate(self, prompt: str):
        # Just return an async generator - Pulsing handles streaming automatically
        for token in self.generate_tokens(prompt):
            yield {"token": token}

# Consume the stream
async for chunk in generator.generate("Hello"):
    print(chunk["token"], end="", flush=True)
```

---

## Supervision (Actor-Level Restart)

Pulsing supports automatic actor restart on failure:

```python
@pul.remote(
    restart_policy="on_failure",  # "never" | "on_failure" | "always"
    max_restarts=3,
    min_backoff=1.0,
    max_backoff=60.0
)
class ReliableWorker:
    def process(self, data):
        # If this crashes, actor restarts automatically
        return heavy_computation(data)
```

!!! note
    Restart restores the actor but **not** its in-memory state. See [Reliability Guide](reliability.md) for idempotency patterns.

---

## Error Handling

Pulsing provides a unified error handling system with clear error categorization.

### Throwing Errors

```python
from pulsing.exceptions import (
    PulsingBusinessError,
    PulsingSystemError,
    PulsingTimeoutError,
)

@pul.remote
class Service:
    async def validate(self, data: str) -> bool:
        if not data:
            raise PulsingBusinessError(400, "Data required")
        return True

    async def process(self, data: str) -> str:
        try:
            return expensive_operation(data)
        except Exception as e:
            raise PulsingSystemError(f"Processing failed: {e}", recoverable=True)

    async def fetch_with_timeout(self, url: str) -> str:
        try:
            return await asyncio.wait_for(httpx.get(url), timeout=5.0)
        except asyncio.TimeoutError:
            raise PulsingTimeoutError("fetch", duration_ms=5000)
```

### Catching Errors

```python
from pulsing.exceptions import (
    PulsingBusinessError,
    PulsingSystemError,
    PulsingRuntimeError,
)

try:
    result = await service.process(data)
except PulsingBusinessError as e:
    # Handle business logic error
    print(f"Validation error: {e.message}")
except PulsingSystemError as e:
    # Handle system error
    if e.recoverable:
        # May retry or wait for actor restart
        pass
    else:
        # Non-recoverable error
        logger.error(f"Fatal error: {e.error}")
except PulsingRuntimeError as e:
    # Handle framework error (network, cluster, etc.)
    print(f"System error: {e}")
```

### Automatic Error Classification

Standard Python exceptions are automatically classified:

```python
@pul.remote
class Processor:
    def process(self, data: str) -> str:
        if not data:
            # ValueError → PulsingBusinessError (code=400)
            raise ValueError("Data required")

        # Other exceptions → PulsingSystemError (recoverable=True)
        return process_data(data)
```

---

## Advanced Patterns

### 1. Stateful Actor

```python
@pul.remote
class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"user_id": user_id, "data": {}}
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        return self.sessions.get(session_id)
```

### 2. Worker Pool (Round-Robin)

```python
@pul.remote
class WorkerPool:
    def __init__(self, workers: list):
        self.workers = workers
        self.idx = 0

    async def submit(self, task: dict):
        worker = self.workers[self.idx]
        self.idx = (self.idx + 1) % len(self.workers)
        return await worker.process(task)
```

### 3. Pipeline

```python
@pul.remote
class PipelineStage:
    def __init__(self, next_stage=None):
        self.next_stage = next_stage

    async def process(self, data: dict) -> dict:
        result = await self.transform(data)
        if self.next_stage:
            return await self.next_stage.process(result)
        return result
```

### 4. LLM Inference Service

```python
@pul.remote
class LLMService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    async def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Best Practices

| ✅ DO | ❌ DON'T |
|-------|----------|
| Single responsibility per actor | Share mutable state between actors |
| Use async for I/O | Block in methods |
| Handle errors gracefully | Ignore exceptions |
| Initialize state in `__init__` | Use global variables |

### Error Handling

```python
@pul.remote
class ResilientActor:
    async def risky_operation(self, data: dict) -> dict:
        try:
            result = await self.process(data)
            return {"success": True, "result": result}
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
```

---

## Quick Reference

### Common Operations

```python
import pulsing as pul

# Create system
system = await pul.actor_system()

# Spawn named actor (discoverable via resolve)
actor = await system.spawn(MyActor(), name="my_actor")

# Call method
result = await actor.ask({"action": "do_something"})

# Using @pul.remote decorator (recommended)
@pul.remote
class MyService:
    def process(self, data): return data

service = await MyService.spawn(name="service")
result = await service.process("hello")

# Resolve existing actor
proxy = await MyService.resolve("service")

# Shutdown
await system.shutdown()
```

---

## Next Steps

- [Remote Actors](remote_actors.md) — cluster communication
- [Reliability](reliability.md) — idempotency, retries, timeouts
- [Operations](operations.md) — CLI tools for inspection
- [LLM Inference](../examples/llm_inference.md) — production inference setup
