# API Overview

Pulsing is a distributed actor framework that provides a communication backbone for building distributed systems and applications.

## Core Concepts

### Actor System

Pulsing is built around the [Actor Model](https://en.wikipedia.org/wiki/Actor_model), where actors are the fundamental units of computation. Actors communicate via asynchronous message passing, providing:

- **Location Transparency**: Same API for local and remote actors
- **Fault Tolerance**: Actors can fail independently without affecting others
- **Concurrency**: Actors process messages one at a time, simplifying concurrent programming

### Key Features

- **Zero External Dependencies**: Pure Rust + Tokio implementation
- **Built-in Service Discovery**: SWIM/Gossip protocol for cluster management
- **Streaming Support**: Native support for streaming requests/responses
- **Multi-Language**: Python-first with Rust core, extensible to other languages

## API Styles

### Python APIs

Pulsing provides multiple API styles to fit different use cases:

#### 1. Actor System Style (Explicit Management)

```python
import pulsing as pul

# Create and manage actor system explicitly
system = await pul.actor_system(addr="0.0.0.0:8000")

# Spawn actors
actor = await system.spawn(MyActor(), name="my_actor")

# Communicate
response = await actor.ask({"message": "hello"})

# Shutdown
await system.shutdown()
```

#### 2. Ray-Style Global API (Convenience)

```python
import pulsing as pul

# Initialize global system
await pul.init(addr="0.0.0.0:8000")

# Spawn actors using global system
actor = await pul.spawn(MyActor(), name="my_actor")

# Communicate
response = await actor.ask({"message": "hello"})

# Shutdown
await pul.shutdown()
```

#### 3. Ray-Compatible API (Migration)

```python
from pulsing.compat import ray

# Ray-compatible API for easy migration
ray.init(address="0.0.0.0:8000")

@ray.remote
class MyActor:
    def process(self, data):
        return f"Processed: {data}"

actor = MyActor.remote()
result = ray.get(actor.process.remote("hello"))

ray.shutdown()
```

### Actor Patterns

#### Remote Decorator (Recommended)

```python
import pulsing as pul

@pul.remote
class Counter:
    def __init__(self, init=0):
        self.value = init

    # Synchronous method - serial execution
    def incr(self):
        self.value += 1
        return self.value

    # Asynchronous method - concurrent execution
    async def fetch_and_add(self, url):
        data = await http_get(url)
        self.value += data
        return self.value

# Usage
counter = await Counter.spawn(name="counter")
result = await counter.incr()
```

#### Base Actor Class

```python
from pulsing.actor import Actor

class MyActor(Actor):
    async def receive(self, msg):
        if msg.get("action") == "greet":
            return f"Hello, {msg.get('name', 'World')}!"
        return "Unknown action"

# Usage
system = await pul.actor_system()
actor = await system.spawn(MyActor(), name="greeter")
response = await actor.ask({"action": "greet", "name": "Alice"})
```

### Message Passing

#### Ask vs Tell

- **`ask(msg)`**: Request/response pattern, waits for and returns a response
- **`tell(msg)`**: Fire-and-forget pattern, sends message without waiting

```python
# Ask - get response
response = await actor.ask({"action": "compute", "data": [1, 2, 3]})

# Tell - no response expected
await actor.tell({"action": "log", "level": "info", "message": "Event occurred"})
```

### Streaming

Pulsing supports streaming responses for large data or continuous generation:

```python
@pul.remote
class StreamingService:
    async def generate_tokens(self, prompt):
        for token in generate_tokens(prompt):
            yield token

# Usage
service = await StreamingService.spawn()
async for token in service.generate_tokens("Hello world"):
    print(token, end="")
```

### Supervision & Fault Tolerance

Actors can be configured with restart policies for fault tolerance:

```python
@pul.remote(
    restart_policy="on_failure",  # "never", "on_failure", "always"
    max_restarts=3,
    min_backoff=0.1,
    max_backoff=30.0
)
class ResilientWorker:
    def process(self, data):
        # If this raises an exception, the actor will be restarted
        return risky_computation(data)
```

### Distributed Queues

Pulsing includes a distributed queue system for data pipelines:

```python
# Writer
writer = await system.queue.write("my_topic", bucket_column="user_id")
await writer.put({"user_id": "u1", "data": "hello"})
await writer.flush()

# Reader
reader = await system.queue.read("my_topic")
records = await reader.get(limit=100)
```

## Rust APIs

### Core Traits

Rust API is organized into trait layers:

#### ActorSystemCoreExt (Primary API)

```rust
use pulsing_actor::prelude::*;

// Spawn actors
let actor = system.spawn_named("services/echo", EchoActor).await?;

// Communicate
let response = actor.ask(Ping(42)).await?;
```

#### Actor Implementation

```rust
use pulsing_actor::prelude::*;
use async_trait::async_trait;

struct MyActor;

#[async_trait]
impl Actor for MyActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        // Process message and return response
        Message::pack(&Pong(42))
    }
}
```

### Behavior (Type-Safe Actors)

```rust
use pulsing_actor::prelude::*;

fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n, _ctx| {
        *count += n;
        BehaviorAction::Same
    })
}

// Usage
let counter = system.spawn(counter(0)).await?;
```

## Error Handling

### Python

```python
try:
    response = await actor.ask({"action": "process", "data": data})
except RuntimeError as e:
    # Actor-side exceptions are wrapped as RuntimeError
    print(f"Actor error: {e}")
except ConnectionError as e:
    # Network errors
    print(f"Connection error: {e}")
except asyncio.TimeoutError as e:
    # Timeout errors
    print(f"Timeout: {e}")
```

### Rust

```rust
use anyhow::Result;

match actor.ask(Ping(42)).await {
    Ok(response) => println!("Got: {:?}", response),
    Err(e) => println!("Error: {:?}", e),
}
```

## Security Considerations

### Trust Boundaries

- **Pickle payloads** in Python-Python communication can lead to RCE if untrusted
- Use TLS in production deployments
- Treat the cluster as an authenticated trust boundary

### Network Security

```python
# Enable TLS
system = await pul.actor_system(
    addr="0.0.0.0:8000",
    passphrase="your-secret-passphrase"
)
```

## Performance Characteristics

- **Low Latency**: HTTP/2 transport with binary serialization
- **High Throughput**: Async runtime with efficient task scheduling
- **Memory Efficient**: Actor-based concurrency without threads
- **Scalable**: Gossip-based cluster discovery for large deployments

## Next Steps

- **[Python API Reference](python.md)**: Complete Python API documentation
- **[Rust API Reference](rust.md)**: Complete Rust API documentation
- **[Examples](../examples/)**: Working code examples
- **[Guide](../guide/)**: In-depth guides and tutorials
