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

from pulsing._core import (
    ActorId,
    ActorRef,
    ActorSystem,
    Message,
    NodeId,
    StreamMessage,
    StreamReader,
    StreamWriter,
    SystemConfig,
)

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

    def on_start(self, actor_id: ActorId) -> None:  # noqa: B027
        """Called when actor starts. Override to handle actor startup."""
        pass

    def on_stop(self) -> None:  # noqa: B027
        """Called when actor stops. Override to handle actor cleanup."""
        pass

    def metadata(self) -> dict[str, str]:
        """Return actor metadata for diagnostics"""
        return {}

    @abstractmethod
    async def receive(self, msg: Message) -> Message | StreamMessage | None:
        """
        Handle incoming message

        Args:
            msg: Incoming message (use msg.to_json() to get data)

        Returns:
            - Message.from_json("Type", {...}): Single response
            - StreamMessage.create(...): Streaming response
            - None: No response

        Example:
            async def receive(self, msg):
                data = msg.to_json()
                if msg.msg_type == "Ping":
                    return Message.from_json("Pong", {"count": 1})
                return None
        """
        pass
