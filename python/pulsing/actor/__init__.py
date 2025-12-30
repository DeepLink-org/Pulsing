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
    SealedPyMessage,
    StreamMessage,
    StreamReader,
    StreamWriter,
    SystemConfig,
)

from . import helpers
from .remote import (
    PYTHON_ACTOR_SERVICE_NAME,
    ActorClass,
    ActorProxy,
    PythonActorService,
    as_actor,
    get_metrics,
    get_node_info,
    health_check,
    list_actors,
    ping,
    remote,
)

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
    "SealedPyMessage",
    # Streaming types
    "StreamReader",
    "StreamWriter",
    # Helper functions
    "create_actor_system",
    "helpers",
    # Actor decorator
    "as_actor",
    "ActorClass",
    "ActorProxy",
    "remote",  # Alias for backward compatibility
    # System helper functions
    "list_actors",
    "get_metrics",
    "get_node_info",
    "health_check",
    "ping",
]


async def create_actor_system(config: SystemConfig) -> ActorSystem:
    """
    Create a new ActorSystem with automatic event loop injection.

    This is a convenience function that wraps ActorSystem.create() to automatically
    inject the current event loop, making it easier to use.

    The function also automatically registers PythonActorService for remote actor creation.

    Args:
        config: SystemConfig instance (use SystemConfig.standalone() or SystemConfig.with_addr())

    Returns:
        ActorSystem instance

    Example:
        config = SystemConfig.with_addr("0.0.0.0:8000")
        system = await create_actor_system(config)
    """
    loop = asyncio.get_running_loop()
    system = await ActorSystem.create(config, loop)

    # Automatically register PythonActorService (for remote actor creation)
    service = PythonActorService(system)
    await system.spawn(PYTHON_ACTOR_SERVICE_NAME, service, public=True)

    return system


class Actor(ABC):
    """Base class for Python actors. Implement `receive` to handle messages.

    Python actors can receive and return arbitrary Python objects when communicating
    with other Python actors. The objects are automatically pickled and unpickled.

    For communication with Rust actors, use Message.from_json() and msg.to_json().
    """

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
    async def receive(self, msg):
        """
        Handle incoming message

        Args:
            msg: Incoming message. Can be:
                 - Any Python object (when called from Python actors with ask/tell)
                 - Message object (when called from Rust actors or with Message.from_json)

        Returns:
            - Any Python object: automatically pickled for Python-to-Python communication
            - Message.from_json("Type", {...}): JSON response for Rust actor communication
            - StreamMessage.create(...): Streaming response
            - None: No response

        Example (Python-to-Python, simple objects):
            # Caller:
            result = await counter.ask({"action": "increment", "n": 10})

            # Actor receive:
            async def receive(self, msg):
                if isinstance(msg, dict) and msg.get("action") == "increment":
                    self.value += msg["n"]
                    return {"value": self.value}

        Example (Rust actor communication):
            async def receive(self, msg):
                if isinstance(msg, Message) and msg.msg_type == "Ping":
                    return Message.from_json("Pong", {"count": 1})
                return None
        """
        pass
