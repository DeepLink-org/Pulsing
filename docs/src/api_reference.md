# API Reference

Complete API documentation for Pulsing Actor Framework.

## Core Classes

### Actor

Base class for all actors.

```python
class Actor:
    async def receive(self, msg: Message) -> Message:
        """Handle incoming messages."""
        pass
```

### Message

Message wrapper for actor communication.

```python
class Message:
    @staticmethod
    def single(msg_type: str, payload: bytes) -> Message:
        """Create a single message."""
        pass

    @staticmethod
    def stream(msg_type: str, payload: bytes) -> Message:
        """Create a streaming message."""
        pass
```

### SystemConfig

Configuration for Actor System.

```python
class SystemConfig:
    @staticmethod
    def standalone() -> SystemConfig:
        """Create standalone (non-cluster) configuration."""
        pass

    @staticmethod
    def with_addr(addr: str) -> SystemConfig:
        """Create configuration with address."""
        pass

    def with_seeds(self, seeds: List[str]) -> SystemConfig:
        """Add seed nodes for cluster discovery."""
        pass
```

### ActorSystem

Main entry point for the actor system.

```python
class ActorSystem:
    async def spawn(
        self,
        actor: Actor,
        name: str,
        public: bool = False
    ) -> ActorRef:
        """Spawn a new actor."""
        pass

    async def find(self, name: str) -> Optional[ActorRef]:
        """Find an actor by name in the cluster."""
        pass

    async def has_actor(self, name: str) -> bool:
        """Check if an actor exists."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the actor system."""
        pass
```

### ActorRef

Reference to an actor (local or remote).

```python
class ActorRef:
    async def ask(self, msg: Message) -> Message:
        """Send a message and wait for response."""
        pass

    async def tell(self, msg: Message) -> None:
        """Send a message without waiting for response."""
        pass

    async def ask_stream(self, msg: Message) -> AsyncIterator[Message]:
        """Send a streaming message."""
        pass
```

## Decorators

### @as_actor

Convert a class into an Actor automatically.

```python
@as_actor
class MyActor:
    def __init__(self, value: int):
        self.value = value

    def get(self) -> int:
        return self.value

    async def process(self, data: str) -> dict:
        return {"result": data.upper()}
```

After decoration, the class provides:

- `local(system, **kwargs) -> ActorRef`: Create actor locally
- `remote(system, **kwargs) -> ActorRef`: Create actor remotely (or locally if single node)

## Functions

### create_actor_system

Create a new Actor System instance.

```python
async def create_actor_system(config: SystemConfig) -> ActorSystem:
    """Create and start an actor system."""
    pass
```

## Examples

See the [Quick Start Guide](quickstart/index.md) for usage examples.
