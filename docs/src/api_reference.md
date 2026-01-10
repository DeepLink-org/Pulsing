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
    @property
    def msg_type(self) -> str:
        """Get the message type."""
        pass

    @property
    def payload(self) -> bytes:
        """Get the raw payload bytes."""
        pass

    @property
    def is_stream(self) -> bool:
        """Check if this is a streaming message."""
        pass

    @staticmethod
    def single(msg_type: str, payload: bytes) -> Message:
        """Create a single message with raw bytes."""
        pass

    def to_json(self) -> Any:
        """Deserialize payload as JSON."""
        pass

    def to_object(self) -> Any:
        """Deserialize payload as Python object (pickle)."""
        pass

    def stream_reader(self) -> StreamReader:
        """Get stream reader for streaming messages."""
        pass
```

### StreamMessage

Factory for creating streaming responses.

```python
class StreamMessage:
    @staticmethod
    def create(
        msg_type: str = "",
        buffer_size: int = 32
    ) -> tuple[Message, StreamWriter]:
        """
        Create a streaming message and its writer.

        Args:
            msg_type: Default message type for stream chunks
            buffer_size: Bounded channel buffer size (backpressure)

        Returns:
            tuple of (Message, StreamWriter)
        """
        pass
```

### StreamWriter

Writer for streaming responses. Supports automatic Python object serialization.

```python
class StreamWriter:
    async def write(self, obj: Any) -> None:
        """
        Write a Python object to the stream.

        The object is automatically serialized using pickle,
        making Python-to-Python streaming transparent.

        Args:
            obj: Any picklable Python object (dict, list, str, etc.)
        """
        pass

    async def close(self) -> None:
        """Close the stream normally."""
        pass

    async def error(self, message: str) -> None:
        """Close the stream with an error."""
        pass
```

### StreamReader

Reader for streaming responses. Automatically deserializes Python objects.

```python
class StreamReader:
    async def __anext__(self) -> Any:
        """
        Get the next item from the stream.

        Returns Python objects directly (automatically unpickled).
        Raises StopAsyncIteration when stream ends.
        """
        pass

    def __aiter__(self) -> StreamReader:
        """Return self as async iterator."""
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

### @remote

Convert a class into an Actor automatically.

```python
from pulsing.actor import init, shutdown, remote

@remote
class MyActor:
    def __init__(self, value: int):
        self.value = value

    def get(self) -> int:
        return self.value

    async def process(self, data: str) -> dict:
        return {"result": data.upper()}

async def main():
    await init()
    actor = await MyActor.spawn(value=10)
    print(await actor.get())  # 10
    await shutdown()
```

After decoration, the class provides:

- `spawn(**kwargs) -> ActorRef`: Create actor (uses global system from `init()`)

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
