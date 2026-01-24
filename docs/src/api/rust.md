# Rust API Reference

This page provides an overview of Pulsing's Rust API with examples and usage patterns.

## Installation

Add Pulsing to your `Cargo.toml`:

```toml
[dependencies]
pulsing-actor = "0.1"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

## Core Concepts

The Rust API is organized into trait layers that provide different levels of functionality:

- **ActorSystemCoreExt**: Primary API for spawning and resolving actors
- **ActorSystemAdvancedExt**: Advanced features like supervision and factory-based spawning
- **ActorSystemOpsExt**: Operations, diagnostics, and lifecycle management

## Quick Start

```rust
use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Ping(i32);

#[derive(Serialize, Deserialize)]
struct Pong(i32);

struct Echo;

#[async_trait::async_trait]
impl Actor for Echo {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let Ping(x) = msg.unpack()?;
        Message::pack(&Pong(x))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    // Spawn a named actor
    let actor = system.spawn_named("services/echo", Echo).await?;

    // Send a message and wait for response
    let Pong(x): Pong = actor.ask(Ping(42)).await?;

    println!("Received: {}", x);

    system.shutdown().await?;
    Ok(())
}
```

## Core API

### ActorSystem

The main entry point for the actor system.

```rust
pub struct ActorSystem { /* fields omitted */ }

impl ActorSystem {
    pub async fn builder() -> ActorSystemBuilder {
        // Create a new actor system builder
    }
}
```

#### ActorSystemBuilder

Builder pattern for configuring the actor system.

```rust
pub struct ActorSystemBuilder { /* fields omitted */ }

impl ActorSystemBuilder {
    pub fn addr<A: Into<String>>(self, addr: A) -> Self {
        // Set the bind address
    }

    pub fn seeds<I: IntoIterator<Item = String>>(self, seeds: I) -> Self {
        // Set seed nodes for cluster discovery
    }

    pub fn build(self) -> impl Future<Output = anyhow::Result<ActorSystem>> {
        // Build the actor system
    }
}
```

### ActorSystemCoreExt

Core spawning and resolving functionality.

```rust
#[async_trait::async_trait]
pub trait ActorSystemCoreExt {
    async fn spawn<A>(&self, actor: A) -> anyhow::Result<TypedRef<A::Message>>
    where
        A: Actor + 'static;

    async fn spawn_named<A>(
        &self,
        name: &str,
        actor: A
    ) -> anyhow::Result<TypedRef<A::Message>>
    where
        A: Actor + 'static;

    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef>;

    async fn resolve(&self, name: &str) -> anyhow::Result<ActorRef>;
}
```

### Actor Trait

The core trait that all actors must implement.

```rust
#[async_trait::async_trait]
pub trait Actor: Send + 'static {
    type Message: Serialize + for<'de> Deserialize<'de> + Send + 'static;

    async fn receive(
        &mut self,
        msg: Message,
        ctx: &mut ActorContext
    ) -> anyhow::Result<Message>;

    fn on_start(&mut self, _id: ActorId, _ctx: &mut ActorContext) {}

    fn on_stop(&mut self, _ctx: &mut ActorContext) {}
}
```

### TypedRef

Type-safe reference to an actor.

```rust
pub struct TypedRef<M> { /* fields omitted */ }

impl<M> TypedRef<M>
where
    M: Serialize + for<'de> Deserialize<'de> + Send + 'static,
{
    pub async fn ask(&self, msg: M) -> anyhow::Result<M> {
        // Send message and wait for typed response
    }

    pub async fn tell(&self, msg: M) -> anyhow::Result<()> {
        // Send message without waiting for response
    }
}
```

## Advanced Features

### Supervision

Actors can be configured with restart policies for fault tolerance.

```rust
use pulsing_actor::system::SupervisionSpec;

let options = SpawnOptions::new()
    .supervision(SupervisionSpec::on_failure().max_restarts(3));

// Factory-based spawning with supervision
system.spawn_named_factory("services/worker", || Ok(Worker::new()), options).await?;
```

### Behavior (Type-Safe Actors)

A higher-level API for type-safe actors using the behavior pattern.

```rust
use pulsing_actor::prelude::*;

fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n, _ctx| {
        *count += n;
        BehaviorAction::Same
    })
}

// Behavior implements IntoActor trait
let counter = system.spawn(counter(0)).await?;
let result: i32 = counter.ask(5).await?; // Result is 5
```

### Message Types

#### Regular Messages

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct MyMessage {
    action: String,
    data: Vec<i32>,
}

// Pack/unpack messages
let msg = Message::pack(&MyMessage {
    action: "process".to_string(),
    data: vec![1, 2, 3],
})?;

let MyMessage { action, data } = msg.unpack()?;
```

#### Streaming Messages

```rust
// For streaming responses
let stream_msg = Message::Stream(Stream::from_iter(items));

// Handle streaming in actor
async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
    match msg {
        Message::Stream(stream) => {
            // Process stream
            let result = process_stream(stream).await?;
            Message::pack(&result)
        }
        _ => Message::pack(&"Unsupported message type")
    }
}
```

## Cluster Management

### Node Discovery

Pulsing uses the SWIM protocol for automatic cluster discovery.

```rust
// Single node
let system = ActorSystem::builder()
    .addr("0.0.0.0:8000")
    .build()
    .await?;

// Join existing cluster
let system = ActorSystem::builder()
    .addr("0.0.0.0:8001")
    .seeds(vec!["127.0.0.1:8000".to_string()])
    .build()
    .await?;
```

### ActorSystemOpsExt

Operations and diagnostics.

```rust
#[async_trait::async_trait]
pub trait ActorSystemOpsExt {
    fn node_id(&self) -> NodeId;

    fn addr(&self) -> &str;

    async fn members(&self) -> Vec<NodeInfo>;

    async fn all_named_actors(&self) -> HashMap<String, ActorId>;

    async fn stop(&self, name: &str) -> anyhow::Result<()>;

    async fn shutdown(self) -> anyhow::Result<()>;
}
```

## Error Handling

Pulsing uses `anyhow::Result<T>` for error handling throughout the API.

```rust
use anyhow::{Result, Context};

async fn my_actor_logic(system: &ActorSystem) -> Result<()> {
    let actor = system.spawn_named("my_actor", MyActor)
        .await
        .context("Failed to spawn actor")?;

    let response = actor.ask(MyMessage::default())
        .await
        .context("Failed to send message")?;

    Ok(())
}
```

## Examples

### HTTP Server Actor

```rust
use pulsing_actor::prelude::*;
use warp::Filter;

struct HttpServer {
    system: ActorSystem,
}

#[async_trait::async_trait]
impl Actor for HttpServer {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        // Handle HTTP requests by forwarding to other actors
        let request: HttpRequest = msg.unpack()?;
        let processor = self.system.resolve("request_processor").await?;
        processor.ask(request).await
    }
}
```

### Worker Pool

```rust
use pulsing_actor::prelude::*;

struct WorkerPool {
    workers: Vec<ActorRef>,
}

#[async_trait::async_trait]
impl Actor for WorkerPool {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        // Round-robin task distribution
        let worker = &self.workers[self.next_worker()];
        worker.ask(msg).await
    }
}
```

## Performance Considerations

- **Zero-copy messaging**: Messages are passed by reference when possible
- **Async runtime**: Built on Tokio for high concurrency
- **Binary serialization**: Efficient bincode serialization
- **Connection pooling**: HTTP/2 connection reuse

## Integration

### With Axum/Warp

```rust
use axum::{routing::post, Router};
use pulsing_actor::prelude::*;

async fn handle_request(
    Extension(system): Extension<ActorSystem>,
    Json(payload): Json<MyRequest>,
) -> Json<MyResponse> {
    let actor = system.resolve("request_handler").await?;
    let response: MyResponse = actor.ask(payload).await?;
    Ok(Json(response))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    let app = Router::new()
        .route("/api", post(handle_request))
        .layer(Extension(system.clone()));

    // Start both HTTP server and actor system
    tokio::select! {
        _ = serve(app, ([127, 0, 0, 1], 3000)) => {},
        _ = system.run() => {},
    }

    Ok(())
}
```

## Next Steps

- **[Python API](python.md)**: Python interface documentation
- **[Examples](../../examples/)**: Working Rust examples
- **[Design Documents](../../design/)**: Architecture and design decisions
