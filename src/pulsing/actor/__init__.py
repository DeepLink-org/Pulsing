"""
Pulsing Actor System - Python bindings for the distributed actor framework.

This module provides a Pythonic interface to the Pulsing Actor System, allowing you to:
- Create and manage ActorSystem instances
- Define actors in Python by implementing the receive method
- Send messages between actors using ask/tell patterns
- Build distributed actor clusters with gossip-based discovery
- Handle streaming requests and responses

Example - Basic Actor:
    ```python
    import asyncio
    from pulsing.actor import ActorSystem, SystemConfig, Message, Actor

    class EchoActor(Actor):
        async def receive(self, msg: Message) -> Message:
            # Echo back the same message
            return msg

    async def main():
        config = SystemConfig.standalone()
        system = await ActorSystem.create(config)
        actor_ref = await system.spawn("echo", EchoActor())
        response = await actor_ref.ask_json("greeting", {"text": "Hello!"})
        print(f"Got response: {response}")
        await system.shutdown()

    asyncio.run(main())
    ```

Example - Streaming Response:
    ```python
    from pulsing.actor import Actor, Message, StreamMessage

    class StreamingActor(Actor):
        async def receive(self, msg: Message) -> Message:
            if msg.msg_type == "Generate":
                # Create streaming response
                stream_msg, writer = StreamMessage.create("TokenStream")
                
                async def produce():
                    for i in range(10):
                        await writer.write_json({"token": f"word_{i}"})
                    writer.close()
                
                asyncio.create_task(produce())
                return stream_msg
            
            return Message.empty()
    ```

Example - Consuming Stream:
    ```python
    class ConsumerActor(Actor):
        async def receive(self, msg: Message) -> Message:
            if msg.is_stream:
                reader = msg.stream_reader()
                results = []
                async for chunk in reader:
                    data = json.loads(chunk)
                    results.append(data)
                return Message.from_json("Result", {"items": results})
            
            return Message.empty()
    ```
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Union, AsyncIterator, Tuple

# Import native bindings
from dynamo._core import actor as _actor_module

NodeId = _actor_module.NodeId
ActorId = _actor_module.ActorId
RawMessage = _actor_module.Message  # Legacy alias
ActorRef = _actor_module.ActorRef
SystemConfig = _actor_module.SystemConfig
_ActorSystem = _actor_module.ActorSystem
# Streaming types
StreamReader = _actor_module.StreamReader
StreamWriter = _actor_module.StreamWriter
StreamMessage = _actor_module.StreamMessage
Message = _actor_module.UnifiedMessage

__all__ = [
    # Core types
    "NodeId",
    "ActorId",
    "ActorRef",
    "SystemConfig",
    "ActorSystem",
    "Actor",
    # Message types
    "Message",
    "RawMessage",  # Legacy alias
    # Streaming types
    "StreamReader",
    "StreamWriter", 
    "StreamMessage",
]


class Actor(ABC):
    """
    Base class for Python actors.
    
    Subclass this and implement the `receive` method to create your own actor.
    The receive method can be either synchronous or asynchronous.
    
    The `receive` method receives a `Message` which can be either:
    - Single message: Access payload with `msg.payload` or `msg.to_json()`
    - Stream message: Check with `msg.is_stream` and use `msg.stream_reader()`
    
    You can return:
    - `Message.single(msg_type, payload)` or `Message.from_json(msg_type, data)` for single response
    - `StreamMessage.create(msg_type)` for streaming response
    - `Message.empty()` for no response
    
    Example - Basic Actor:
        ```python
        class CounterActor(Actor):
            def __init__(self):
                self.count = 0
            
            def on_start(self, actor_id: ActorId):
                print(f"Actor {actor_id} started")
            
            async def receive(self, msg: Message) -> Message:
                data = msg.to_json()
                if msg.msg_type == "increment":
                    self.count += data.get("value", 1)
                    return Message.from_json("result", {"count": self.count})
                elif msg.msg_type == "get":
                    return Message.from_json("result", {"count": self.count})
                else:
                    return Message.empty()
        ```
    
    Example - Streaming Response:
        ```python
        class GeneratorActor(Actor):
            async def receive(self, msg: Message) -> Message:
                if msg.msg_type == "Generate":
                    stream_msg, writer = StreamMessage.create("Tokens")
                    
                    async def produce():
                        for i in range(10):
                            await writer.write_json({"token": i})
                        writer.close()
                    
                    asyncio.create_task(produce())
                    return stream_msg
                
                return Message.empty()
        ```
    
    Example - Consuming Stream:
        ```python
        class AggregatorActor(Actor):
            async def receive(self, msg: Message) -> Message:
                if msg.is_stream:
                    reader = msg.stream_reader()
                    items = []
                    async for chunk in reader:
                        items.append(json.loads(chunk))
                    return Message.from_json("Result", {"items": items})
                
                return Message.empty()
        ```
    """
    
    def on_start(self, actor_id: ActorId) -> None:
        """Called when the actor starts. Override to add initialization logic."""
        pass
    
    def on_stop(self) -> None:
        """Called when the actor stops. Override to add cleanup logic."""
        pass
    
    def metadata(self) -> Dict[str, str]:
        """
        Return actor metadata for diagnostics.
        
        Returns:
            A dictionary with string keys and string values.
            The transport layer handles serialization.
        """
        return {}

    @abstractmethod
    def receive(self, msg: Message) -> Union[Message, StreamMessage]:
        """
        Handle an incoming message.
        
        This method can be either synchronous or asynchronous (async def).
        
        Args:
            msg: The incoming Message. Check `msg.is_stream` to determine type:
                - Single: Access `msg.payload` or `msg.to_json()`
                - Stream: Use `msg.stream_reader()` to get an async iterator
            
        Returns:
            Response message. Can be:
            - `Message.single(type, data)` or `Message.from_json(type, obj)` for single response
            - `StreamMessage.create(type)` returns (stream_msg, writer) for streaming
            - `Message.empty()` for no response
        """
        pass


class ActorSystem:
    """
    The Actor System - manages actors and cluster membership.
    
    This is a high-level wrapper around the native ActorSystem that provides
    a more Pythonic interface.
    
    Example:
        ```python
        # Standalone mode
        config = SystemConfig.standalone()
        system = await ActorSystem.create(config)
        
        # With specific address
        config = SystemConfig.with_addr("0.0.0.0:8000")
        system = await ActorSystem.create(config)
        
        # Cluster mode with seed nodes
        config = SystemConfig.with_addr("0.0.0.0:8001").with_seeds([
            "192.168.1.100:8000"
        ])
        system = await ActorSystem.create(config)
        ```
    """
    
    def __init__(self, inner: _ActorSystem):
        self._inner = inner
    
    @classmethod
    async def create(cls, config: SystemConfig) -> "ActorSystem":
        """
        Create a new actor system.
        
        Args:
            config: SystemConfig specifying the system configuration
            
        Returns:
            A new ActorSystem instance
        """
        loop = asyncio.get_event_loop()
        inner = await _ActorSystem.create(config, loop)
        return cls(inner)
    
    @property
    def node_id(self) -> NodeId:
        """Get the local node ID."""
        return self._inner.node_id
    
    @property
    def addr(self) -> str:
        """Get the system address."""
        return self._inner.addr
    
    async def spawn(self, name: str, handler: Actor, public: bool = False) -> ActorRef:
        """
        Spawn a new actor.
        
        Args:
            name: Actor name (must be unique within this node)
            handler: Actor instance with a receive method
            public: Whether to broadcast this actor's existence to the cluster (default: False).
                   Set to True only for named services that need to be discoverable by name.
            
        Returns:
            ActorRef to the spawned actor
        """
        return await self._inner.spawn(name, handler, public)
    
    async def actor_ref(self, actor_id: ActorId) -> ActorRef:
        """
        Get a reference to an actor (local or remote).
        
        Args:
            actor_id: The ActorId of the target actor
            
        Returns:
            ActorRef to the actor
            
        Raises:
            Exception: If the actor is not found
        """
        return await self._inner.actor_ref(actor_id)
    
    async def members(self) -> List[Dict[str, str]]:
        """
        Get cluster members.
        
        Returns:
            List of member info dictionaries with keys:
            - node_id: The node's unique ID
            - addr: The node's address
            - status: Member status (Alive, Suspect, Dead)
        """
        return await self._inner.members()
    
    def local_actor_names(self) -> List[str]:
        """Get names of all local actors."""
        return self._inner.local_actor_names()
    
    async def get_named_instances(self, name: str) -> List[Dict[str, str]]:
        """
        Get all instances of a named actor across the cluster.
        
        This is useful for discovering all workers registered with the same name.
        
        Args:
            name: The actor name (e.g., "worker")
            
        Returns:
            List of instance info dictionaries with keys:
            - node_id: The node's unique ID
            - addr: The node's address
            - status: Member status (Alive, Suspect, Dead)
        
        Example:
            ```python
            # Find all workers in the cluster
            workers = await system.get_named_instances("worker")
            for w in workers:
                print(f"Worker at {w['addr']}")
            ```
        """
        return await self._inner.get_named_instances(name)
    
    async def resolve_named(self, name: str) -> ActorRef:
        """
        Resolve a named actor reference (load balanced).
        
        If multiple instances exist, one will be selected using load balancing.
        
        Args:
            name: The actor name (e.g., "worker")
            
        Returns:
            ActorRef to one of the instances
        
        Example:
            ```python
            # Get a reference to one worker (load balanced)
            worker_ref = await system.resolve_named("worker")
            response = await worker_ref.ask_json("Generate", {"prompt": "Hello"})
            ```
        """
        return await self._inner.resolve_named(name)
    
    async def stop(self, actor_name: str) -> None:
        """
        Stop an actor.
        
        Args:
            actor_name: Name of the actor to stop
        """
        await self._inner.stop(actor_name)
    
    async def shutdown(self) -> None:
        """Shutdown the entire actor system."""
        await self._inner.shutdown()
    
    def __repr__(self) -> str:
        return f"ActorSystem(node_id='{self.node_id}', addr='{self.addr}')"

