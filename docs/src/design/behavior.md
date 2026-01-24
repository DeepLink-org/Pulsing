# Behavior API (Rust)

Pulsing provides a type-safe, functional actor programming interface inspired by Akka Typed. The Behavior API offers an alternative to the traditional `Actor` trait with compile-time message type checking.

## Design Philosophy

**Type safety first.**

- `TypedRef<M>` ensures messages are type-checked at compile time
- State transitions via `BehaviorAction::Become` enable clean state machine patterns
- Functional style: actors are just message-handling functions

## Core Concepts

### Behavior&lt;M&gt;

An actor is defined as a function that handles messages of type `M`:

```rust
use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};

fn counter(initial: i32) -> Behavior<i32> {
    stateful(initial, |count, n, _ctx| {
        *count += n;
        println!("count = {}", *count);
        BehaviorAction::Same
    })
}
```

### TypedRef&lt;M&gt;

A type-safe actor reference that only accepts messages of type `M`:

```rust
// Behavior implements IntoActor, can be spawned directly
let counter = system.spawn_named("actors/counter", counter(0)).await?;

counter.tell(5).await?;  // OK
counter.tell(3).await?;  // OK
// counter.tell("hello").await?;  // Runtime error (type mismatch on deserialization)
```

### BehaviorAction

Controls actor lifecycle and state transitions:

| Action | Description |
|--------|-------------|
| `Same` | Keep current behavior |
| `Become(behavior)` | Switch to a new behavior (state machine transition) |
| `Stop(reason)` | Stop the actor gracefully |

## Creating Behaviors

### Stateful Behavior

Use `stateful()` for actors with internal state:

```rust
use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};

fn counter(initial: i32) -> Behavior<i32> {
    stateful(initial, |count, msg, ctx| {
        *count += msg;
        println!("[{}] count = {}", ctx.name(), *count);
        BehaviorAction::Same
    })
}
```

The handler receives:
- `&mut S` - mutable reference to state
- `M` - the message
- `&BehaviorContext<M>` - actor context

### Stateless Behavior

Use `stateless()` for actors without internal state:

```rust
use pulsing_actor::behavior::{stateless, Behavior, BehaviorAction};

fn echo() -> Behavior<String> {
    stateless(|msg, ctx| {
        Box::pin(async move {
            println!("[{}] Received: {}", ctx.name(), msg);
            BehaviorAction::Same
        })
    })
}
```

## State Machine Pattern

Use `BehaviorAction::Become` to implement state machines:

```rust
use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
enum Signal {
    Next,
    Query,
}

// State with shared data
#[derive(Clone)]
struct Stats {
    cycles: u32,
    transitions: u32,
}

fn red(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            println!("[{}] 🔴 Red -> 🟢 Green", ctx.name());
            BehaviorAction::Become(green(stats.clone()))
        }
        Signal::Query => {
            println!("[{}] Current: 🔴 Red", ctx.name());
            BehaviorAction::Same
        }
    })
}

fn green(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            println!("[{}] 🟢 Green -> 🟡 Yellow", ctx.name());
            BehaviorAction::Become(yellow(stats.clone()))
        }
        Signal::Query => {
            println!("[{}] Current: 🟢 Green", ctx.name());
            BehaviorAction::Same
        }
    })
}

fn yellow(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            stats.cycles += 1;
            println!("[{}] 🟡 Yellow -> 🔴 Red (cycle #{})", ctx.name(), stats.cycles);
            BehaviorAction::Become(red(stats.clone()))
        }
        Signal::Query => {
            println!("[{}] Current: 🟡 Yellow", ctx.name());
            BehaviorAction::Same
        }
    })
}
```

## BehaviorContext

The context provides:

```rust
// Get actor's name
let name = ctx.name();

// Get typed self-reference (for reply-to patterns)
let self_ref: TypedRef<M> = ctx.self_ref();

// Get typed reference to another actor
let other: TypedRef<OtherMsg> = ctx.typed_ref("other_actor");

// Schedule a message to self after delay
ctx.schedule_self(msg, Duration::from_secs(5));

// Check if actor should stop
if ctx.is_cancelled() { ... }

// Access the underlying ActorSystem
let system = ctx.system();
```

## TypedRef Operations

```rust
// Fire-and-forget
counter.tell(5).await?;

// Request-response
let result: i32 = counter.ask(CounterMsg::Get).await?;

// With timeout
let result: i32 = counter.ask_timeout(msg, Duration::from_secs(5)).await?;

// Check if actor is alive
if counter.is_alive() { ... }

// Get underlying untyped ActorRef
let actor_ref = counter.as_untyped()?;
```

## Complete Example

```rust
use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};
use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
enum CounterMsg {
    Increment(i32),
    Decrement(i32),
    Get,
}

fn counter(initial: i32) -> Behavior<CounterMsg> {
    stateful(initial, |count, msg, ctx| match msg {
        CounterMsg::Increment(n) => {
            *count += n;
            println!("[{}] +{} = {}", ctx.name(), n, *count);
            BehaviorAction::Same
        }
        CounterMsg::Decrement(n) => {
            *count -= n;
            println!("[{}] -{} = {}", ctx.name(), n, *count);
            BehaviorAction::Same
        }
        CounterMsg::Get => {
            println!("[{}] current = {}", ctx.name(), *count);
            BehaviorAction::Same
        }
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    // Behavior implements IntoActor, can be spawned directly
    let counter = system.spawn_named("actors/counter", counter(0)).await?;

    // Message sending
    counter.tell(CounterMsg::Increment(5)).await?;
    counter.tell(CounterMsg::Increment(3)).await?;
    counter.tell(CounterMsg::Decrement(2)).await?;
    counter.tell(CounterMsg::Get).await?;

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    system.shutdown().await
}
```

## Actor Trait vs Behavior

| Feature | Actor Trait | Behavior API |
|---------|-------------|--------------|
| Type safety | Runtime (Message type) | Compile-time (TypedRef&lt;M&gt;) |
| State | Struct fields | Encapsulated in closure |
| State machine | Manual | `BehaviorAction::Become` |
| Style | OOP (impl trait) | Functional (functions) |
| Flexibility | Higher | Structured |

**When to use Behavior:**

- Need compile-time message type checking
- Building state machines
- Prefer functional programming style

**When to use Actor trait:**

- Need maximum flexibility
- Complex initialization logic
- Existing OOP codebase

## Running Examples

```bash
# Counter example
cargo run --example behavior_counter -p pulsing-actor

# State machine example
cargo run --example behavior_fsm -p pulsing-actor
```

## What's Next?

- [Actor System](actor-system.md) — Core actor infrastructure
- [Architecture](architecture.md) — System design overview
