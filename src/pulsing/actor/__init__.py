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
from typing import Optional, Dict, Union

from dynamo._core import actor as _actor_module

# 直接导出Rust类型
ActorSystem = _actor_module.ActorSystem
NodeId = _actor_module.NodeId
ActorId = _actor_module.ActorId
ActorRef = _actor_module.ActorRef
SystemConfig = _actor_module.SystemConfig
StreamReader = _actor_module.StreamReader
StreamWriter = _actor_module.StreamWriter
StreamMessage = _actor_module.StreamMessage
Message = _actor_module.UnifiedMessage

from . import helpers

__all__ = [
    # Core types
    "ActorSystem",
    "NodeId",
    "ActorId",
    "ActorRef",
    "SystemConfig",
    "Actor",
    # Message types
    "Message",
    "StreamMessage",
    # Streaming types
    "StreamReader",
    "StreamWriter",
    # Helper functions
    "create_actor_system",
    "helpers",
]


async def create_actor_system(config: SystemConfig) -> ActorSystem:
    """
    Create a new ActorSystem with automatic event loop injection.
    
    This is a convenience function that wraps ActorSystem.create() to automatically
    inject the current event loop, making it easier to use.
    
    Args:
        config: SystemConfig instance (use SystemConfig.standalone() or SystemConfig.with_addr())
    
    Returns:
        ActorSystem instance
        
    Example:
        config = SystemConfig.with_addr("0.0.0.0:8000")
        system = await create_actor_system(config)
    """
    loop = asyncio.get_running_loop()
    return await ActorSystem.create(config, loop)


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
    def receive(self, msg: Message) -> Optional[Union[Message, StreamMessage]]:
        """
        Handle incoming message
        
        Args:
            msg: Incoming message (use msg.to_json() to get data)
            
        Returns:
            - Message.from_json("Type", {...}): Single response
            - StreamMessage.create(...): Streaming response
            - None: No response
            
        Example:
            def receive(self, msg):
                data = msg.to_json()
                if msg.msg_type == "Ping":
                    return Message.from_json("Pong", {"count": 1})
                return None
        """
        pass

