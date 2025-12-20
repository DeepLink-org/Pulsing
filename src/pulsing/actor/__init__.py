"""
Pulsing Actor System - Python bindings for the distributed actor framework.

This module provides a Pythonic interface to the Pulsing Actor System, allowing you to:
- Create and manage ActorSystem instances
- Define actors in Python by implementing the receive method
- Send messages between actors using ask/tell patterns
- Build distributed actor clusters with gossip-based discovery

Example:
    ```python
    import asyncio
    from pulsing.actor import ActorSystem, SystemConfig, RawMessage, Actor

    class EchoActor(Actor):
        async def receive(self, msg: RawMessage) -> RawMessage:
            # Echo back the same message
            return msg

    async def main():
        # Create actor system
        config = SystemConfig.standalone()
        system = await ActorSystem.create(config)

        # Spawn actor
        actor_ref = await system.spawn("echo", EchoActor())

        # Send message
        response = await actor_ref.ask_json("greeting", {"text": "Hello!"})
        print(f"Got response: {response}")

        await system.shutdown()

    asyncio.run(main())
    ```
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

# Import native bindings
from dynamo._core.actor import (
    NodeId,
    ActorId,
    RawMessage,
    ActorRef,
    SystemConfig,
    ActorSystem as _ActorSystem,
)

__all__ = [
    "NodeId",
    "ActorId",
    "RawMessage",
    "ActorRef",
    "SystemConfig",
    "ActorSystem",
    "Actor",
]


class Actor(ABC):
    """
    Base class for Python actors.
    
    Subclass this and implement the `receive` method to create your own actor.
    The receive method can be either synchronous or asynchronous.
    
    Example:
        ```python
        class CounterActor(Actor):
            def __init__(self):
                self.count = 0
            
            def on_start(self, actor_id: ActorId):
                print(f"Actor {actor_id} started")
            
            async def receive(self, msg: RawMessage) -> RawMessage:
                data = msg.to_json()
                if msg.msg_type == "increment":
                    self.count += data.get("value", 1)
                    return RawMessage.from_json("result", {"count": self.count})
                elif msg.msg_type == "get":
                    return RawMessage.from_json("result", {"count": self.count})
                else:
                    return RawMessage.empty()
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
    def receive(self, msg: RawMessage) -> RawMessage:
        """
        Handle an incoming message.
        
        This method can be either synchronous or asynchronous (async def).
        
        Args:
            msg: The incoming RawMessage containing msg_type and payload
            
        Returns:
            A RawMessage response. Use RawMessage.empty() for no response.
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

