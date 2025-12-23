"""
Pulsing Actor System - Python bindings for distributed actor framework

Provides:
- ActorSystem: Manage actors and cluster membership
- Actor: Base class for implementing actors
- Message/StreamMessage: Single and streaming message types
- ActorRef: Reference to local or remote actors
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Union, AsyncIterator, Tuple

from dynamo._core import actor as _actor_module

NodeId = _actor_module.NodeId
ActorId = _actor_module.ActorId
RawMessage = _actor_module.Message
ActorRef = _actor_module.ActorRef
SystemConfig = _actor_module.SystemConfig
_ActorSystem = _actor_module.ActorSystem
StreamReader = _actor_module.StreamReader
StreamWriter = _actor_module.StreamWriter
StreamMessage = _actor_module.StreamMessage
Message = _actor_module.UnifiedMessage

from . import helpers

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
    # Helper module
    "helpers",
]


class Actor(ABC):
    """Base class for Python actors. Implement `receive` to handle messages."""
    
    def on_start(self, actor_id: ActorId) -> None:
        """Called when actor starts"""
        pass
    
    def on_stop(self) -> None:
        """Called when actor stops"""
        pass
    
    def metadata(self) -> Dict[str, str]:
        """Return actor metadata for diagnostics"""
        return {}

    @abstractmethod
    def receive(self, msg: Message) -> Union[Message, StreamMessage]:
        """
        Handle incoming message
        
        Args:
            msg: Single message (use msg.to_json()) or stream (use msg.stream_reader())
            
        Returns:
            Message.from_json() for single, StreamMessage.create() for stream, or Message.empty()
        """
        pass


class ActorSystem:
    """Actor System - manages actors and cluster membership"""
    
    def __init__(self, inner: _ActorSystem):
        self._inner = inner
    
    @classmethod
    async def create(cls, config: SystemConfig) -> "ActorSystem":
        """Create new actor system"""
        loop = asyncio.get_event_loop()
        inner = await _ActorSystem.create(config, loop)
        return cls(inner)
    
    @property
    def node_id(self) -> NodeId:
        return self._inner.node_id
    
    @property
    def addr(self) -> str:
        return self._inner.addr
    
    async def spawn(self, name: str, handler: Actor, public: bool = False) -> ActorRef:
        """Spawn new actor. Set public=True for named services discoverable by name."""
        return await self._inner.spawn(name, handler, public)
    
    async def actor_ref(self, actor_id: ActorId) -> ActorRef:
        """Get reference to actor (local or remote)"""
        return await self._inner.actor_ref(actor_id)
    
    async def members(self) -> List[Dict[str, str]]:
        """Get cluster members with node_id, addr, status"""
        return await self._inner.members()
    
    def local_actor_names(self) -> List[str]:
        """Get names of all local actors"""
        return self._inner.local_actor_names()
    
    async def get_named_instances(self, name: str) -> List[Dict[str, str]]:
        """Get all instances of named actor across cluster"""
        return await self._inner.get_named_instances(name)
    
    async def resolve_named(self, name: str, node_id: Optional[str] = None) -> ActorRef:
        """Resolve named actor reference (load balanced or specific node)"""
        return await self._inner.resolve_named(name, node_id)
    
    async def stop(self, actor_name: str) -> None:
        """Stop an actor"""
        await self._inner.stop(actor_name)
    
    async def shutdown(self) -> None:
        """Shutdown entire actor system"""
        await self._inner.shutdown()
    
    def __repr__(self) -> str:
        return f"ActorSystem(node_id='{self.node_id}', addr='{self.addr}')"

