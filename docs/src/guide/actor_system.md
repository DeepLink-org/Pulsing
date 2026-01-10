# Actor System Guide

This guide covers the core concepts and usage of the Pulsing Actor System.

## What is an Actor?

An Actor is a computational unit that:
- Encapsulates state and behavior
- Communicates only through messages
- Processes messages asynchronously
- Has a unique identifier

## Creating Actors

### Using the Actor Base Class

```python
from pulsing.actor import Actor, Message, SystemConfig, create_actor_system

class MyActor(Actor):
    def __init__(self):
        self.state = {}

    async def receive(self, msg: Message) -> Message:
        # Handle messages
        return Message.single("response", b"data")

async def main():
    system = await create_actor_system(SystemConfig.standalone())
    actor_ref = await system.spawn(MyActor(), "my-actor")
    await system.shutdown()
```

### Using @remote Decorator

The `@remote` decorator automatically converts a class into an Actor:

```python
from pulsing.actor import init, shutdown, remote

@remote
class Counter:
    def __init__(self, value: int = 0):
        self.value = value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

async def main():
    await init()
    counter = await Counter.spawn(value=10)
    print(await counter.get())  # 10
    await shutdown()
```

## Message Passing

### Ask Pattern (Request-Response)

```python
response = await actor_ref.ask(Message.single("ping", b"data"))
```

### Tell Pattern (Fire-and-Forget)

```python
await actor_ref.tell(Message.single("notify", b"data"))
```

### Streaming Messages

```python
msg = Message.stream("process", b"data")
async for chunk in actor_ref.ask_stream(msg):
    print(f"Received: {chunk}")
```

## Actor Lifecycle

Actors have a simple lifecycle:
1. **Creation**: Actor is spawned in the system
2. **Active**: Actor processes messages
3. **Stopped**: Actor is removed from the system

```python
# Create actor
actor_ref = await system.spawn(MyActor(), "my-actor")

# Actor is now active and can receive messages
response = await actor_ref.ask(msg)

# Stop actor
await system.stop("my-actor")
```

## State Management

Actors encapsulate state. Each actor instance has its own isolated state:

```python
@remote
class StatefulActor:
    def __init__(self):
        self.counter = 0
        self.data = {}

    def increment(self) -> int:
        self.counter += 1
        return self.counter

    def set_data(self, key: str, value: str) -> None:
        self.data[key] = value

    def get_data(self, key: str):
        return self.data.get(key)
```

## Error Handling

Errors in actor message handling are propagated to the caller:

```python
try:
    response = await actor_ref.ask(msg)
except Exception as e:
    print(f"Actor error: {e}")
```

## Best Practices

1. **Keep actors focused**: Each actor should have a single responsibility
2. **Use immutable data**: Prefer immutable data structures when possible
3. **Handle errors gracefully**: Always handle potential errors in message processing
4. **Use async methods**: For I/O operations, use async methods
5. **Clean shutdown**: Always call `system.shutdown()` when done

## Next Steps

- Learn about [Remote Actors](remote_actors.md) for cluster communication
- Check the [Design Documents](../design/actor-system.md) for implementation details
