# API Reference

Complete API documentation for Pulsing Actor Framework.

## Core Functions

### pul.actor_system

Create a new Actor System instance.

```python
import pulsing as pul

system = await pul.actor_system(
    addr: str | None = None,        # Bind address, None for standalone
    *,
    seeds: list[str] | None = None, # Seed nodes for cluster
    passphrase: str | None = None,  # TLS passphrase
) -> ActorSystem
```

**Example:**

```python
# Standalone mode
system = await pul.actor_system()

# Cluster mode
system = await pul.actor_system(addr="0.0.0.0:8000")

# Join existing cluster
system = await pul.actor_system(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])

# Shutdown
await system.shutdown()
```

### pul.init / pul.shutdown

Global system initialization (Ray-style async API).

```python
import pulsing as pul

# Initialize global system
await pul.init(addr=None, seeds=None, passphrase=None)

# Use global system
actor = await pul.spawn(MyActor())
ref = await pul.resolve("actor_name")

# Shutdown
await pul.shutdown()
```

## Core Classes

### ActorSystem

Main entry point for the actor system.

```python
class ActorSystem:
    async def spawn(
        self,
        actor: Actor,
        *,
        name: str | None = None,
        public: bool = False,
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0
    ) -> ActorRef:
        """Spawn a new actor."""
        pass

    async def refer(self, actorid: ActorId | str) -> ActorRef:
        """Get ActorRef by ActorId."""
        pass

    async def resolve(self, name: str, *, node_id: int | None = None) -> ActorRef:
        """Resolve actor by name."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the actor system."""
        pass
```

### ActorRef

Low-level reference to an actor. Use `ask()` and `tell()` to communicate.

```python
class ActorRef:
    @property
    def actor_id(self) -> ActorId:
        """Get the actor's ID."""
        pass

    async def ask(self, msg: Any) -> Any:
        """Send a message and wait for response."""
        pass

    async def tell(self, msg: Any) -> None:
        """Send a message without waiting for response (fire-and-forget)."""
        pass
```

### ActorProxy

High-level proxy for `@remote` classes. Call methods directly.

```python
class ActorProxy:
    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef."""
        pass

    # Call methods directly:
    # result = await proxy.my_method(arg1, arg2)
```

## Decorators

### @remote / @pul.remote

Convert a class into a distributed Actor.

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    # Sync method - sequential execution
    def incr(self) -> int:
        self.value += 1
        return self.value

    # Async method - concurrent execution during await
    async def fetch_and_add(self, url: str) -> int:
        data = await http_get(url)
        self.value += data
        return self.value

    # Generator - automatic streaming
    async def stream(self):
        for i in range(10):
            yield {"count": i}

# Create actor
counter = await Counter.spawn(name="counter")

# Call methods directly
result = await counter.incr()

# Streaming
async for chunk in counter.stream():
    print(chunk)

# Resolve existing actor
proxy = await Counter.resolve("counter")
```

**Supervision parameters:**

```python
@pul.remote(
    restart_policy="on_failure",  # "never" | "on_failure" | "always"
    max_restarts=3,
    min_backoff=0.1,
    max_backoff=30.0,
)
class ResilientWorker:
    def work(self, data): ...
```

## Base Actor

For low-level control, inherit from Actor base class.

```python
class MyActor:
    def __init__(self):
        self.value = 0

    def on_start(self, actor_id):
        """Called when actor starts."""
        print(f"Started: {actor_id}")

    async def receive(self, msg):
        """Handle incoming messages."""
        if msg.get("action") == "add":
            self.value += msg.get("n", 1)
            return {"value": self.value}
        return {"error": "unknown action"}

# Spawn
system = await pul.actor_system()
actor = await system.spawn(MyActor(), name="my_actor")

# Communicate via ask/tell
response = await actor.ask({"action": "add", "n": 10})
```

## Queue API

Distributed queue for data pipelines.

```python
# Write
writer = await system.queue.write(
    topic="my_queue",
    bucket_column="user_id",
    num_buckets=4,
)
await writer.put({"user_id": "u1", "data": "hello"})
await writer.flush()

# Read
reader = await system.queue.read("my_queue")
records = await reader.get(limit=100)
```

## Ray Compatibility

Drop-in replacement for Ray.

```python
from pulsing.compat import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    def incr(self):
        self.value += 1
        return self.value

counter = Counter.remote()
result = ray.get(counter.incr.remote())

ray.shutdown()
```

## Examples

See the [Quick Start Guide](quickstart/index.md) for usage examples.
