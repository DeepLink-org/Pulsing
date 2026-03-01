# API Overview

Pulsing is the backbone for distributed AI systems — a distributed actor runtime with streaming, zero dependencies, and built-in discovery.

## Core Concepts

Pulsing is built around the [Actor Model](https://en.wikipedia.org/wiki/Actor_model), where actors are the fundamental units of computation. Actors communicate via asynchronous message passing, providing:

- **Location Transparency**: Same API for local and remote actors
- **Fault Tolerance**: Actors can fail independently without affecting others
- **Concurrency**: Actors process messages one at a time, simplifying concurrent programming

### Key Features

- **Zero External Dependencies**: Pure Rust + Tokio implementation
- **Built-in Service Discovery**: SWIM/Gossip protocol for cluster management
- **Streaming Support**: Native support for streaming requests/responses
- **Multi-Language**: Python-first with Rust core, extensible to other languages

## Quick Start

```python
import pulsing as pul

await pul.init()

@pul.remote
class Counter:
    def __init__(self): self.value = 0
    def incr(self): self.value += 1; return self.value

counter = await Counter.spawn(name="counter")
print(await counter.incr())  # 1

counter2 = await Counter.resolve("counter")
print(await counter2.incr())  # 2

await pul.shutdown()
```

## Python API

You must call `await pul.init()` before using `spawn`, `resolve`, or other APIs.

### Lifecycle

```python
import pulsing as pul

await pul.init(
    addr=None,          # Bind address, None for standalone
    seeds=None,         # Seed nodes for cluster
    passphrase=None,    # TLS passphrase
)

await pul.shutdown()
```

### Define Actor

Use `@pul.remote` to turn any class into a distributed actor:

```python
@pul.remote
class Counter:
    def __init__(self, init=0):
        self.value = init

    def incr(self):                       # sync method — serial execution
        self.value += 1
        return self.value

    async def fetch_and_add(self, url):   # async method — concurrent during await
        data = await http_get(url)
        self.value += data
        return self.value
```

### Create and Call

`Class.spawn()` creates an actor and returns a typed proxy:

```python
counter = await Counter.spawn(name="counter", init=10)
result = await counter.incr()             # direct method call
```

### Resolve Existing Actor

```python
# Typed proxy — when you know the class
proxy = await pul.resolve("counter", cls=Counter, timeout=30)
result = await proxy.incr()

# Or via ActorClass (same result)
proxy = await Counter.resolve("counter")
result = await proxy.incr()

# Untyped proxy — when remote type is unknown (any method call)
proxy = await pul.resolve("service_name")
result = await proxy.any_method(args)
```

### Streaming

Return a generator for streaming responses:

```python
@pul.remote
class StreamingService:
    async def generate_tokens(self, prompt):
        for token in generate_tokens(prompt):
            yield token

service = await StreamingService.spawn()
async for token in service.generate_tokens("Hello world"):
    print(token, end="")
```

### Supervision

```python
@pul.remote(
    restart_policy="on_failure",  # "never", "on_failure", "always"
    max_restarts=3,
    min_backoff=0.1,
    max_backoff=30.0,
)
class ResilientWorker:
    def process(self, data):
        return risky_computation(data)
```

### Queue

Distributed queue with bucket-based partitioning:

```python
writer = await pul.queue.write("my_queue", bucket_column="user_id")
await writer.put({"user_id": "u1", "data": "hello"})
await writer.flush()

reader = await pul.queue.read("my_queue")
records = await reader.get(limit=100)
```

### Topic

Lightweight pub/sub for real-time messaging:

```python
writer = await pul.topic.write("events")
await writer.publish({"type": "user_login", "user": "alice"})

reader = await pul.topic.read("events")

@reader.on_message
async def handle(msg):
    print(f"Received: {msg}")

await reader.start()
```

### Under the Hood

#### ActorSystem (Explicit Management)

```python
import pulsing as pul

system = await pul.actor_system(addr="0.0.0.0:8000")

class MyActor:
    async def receive(self, msg):
        return f"echo: {msg}"

actor = await system.spawn(MyActor(), name="my_actor")
response = await actor.ask({"message": "hello"})
await actor.tell({"event": "fire_and_forget"})

await system.shutdown()
```

---

## Rust API

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

let counter = system.spawn(counter(0)).await?;
```

---

## Error Handling

### Python

```python
from pulsing.exceptions import (
    PulsingBusinessError,
    PulsingSystemError,
    PulsingRuntimeError,
)

try:
    result = await service.process(data)
except PulsingBusinessError as e:
    print(f"Business error [{e.code}]: {e.message}")
except PulsingSystemError as e:
    print(f"System error: {e.error}, recoverable: {e.recoverable}")
except PulsingRuntimeError as e:
    print(f"Framework error: {e}")
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

- **Pickle payloads** in Python-Python communication can lead to RCE if untrusted
- Use TLS in production deployments
- Treat the cluster as an authenticated trust boundary

```python
await pul.init(addr="0.0.0.0:8000", passphrase="your-secret-passphrase")
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
