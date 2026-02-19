"""
Pulsing Core - Python bindings for distributed actor framework

Simple API:
    from pulsing.core import init, shutdown, remote

    await init()

    @remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = await Counter.spawn(init=10)
    result = await counter.incr()

    await shutdown()

Advanced API:
    from pulsing.core import ActorSystem, Actor, Message, SystemConfig
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from pulsing._core import (
    ActorId,
    ActorRef,
    ActorSystem,
    NodeId,
    SealedPyMessage,
    ZeroCopyDescriptor,
    StreamReader,
    StreamWriter,
    SystemConfig,
)
from .messaging import Message, StreamMessage


# =============================================================================
# Global system for simple API
# =============================================================================

_global_system: ActorSystem = None


async def init(
    addr: str = None,
    *,
    seeds: list[str] = None,
    passphrase: str = None,
    head_addr: str = None,
    is_head_node: bool = False,
) -> ActorSystem:
    """Initialize Pulsing actor system

    Args:
        addr: Bind address (e.g., "0.0.0.0:8000"). None for standalone mode.
        seeds: Seed nodes to join cluster (Gossip mode).
        passphrase: Enable TLS with this passphrase.
        head_addr: Address of head node (worker mode). Mutually exclusive with is_head_node.
        is_head_node: If True, this node runs as head. Mutually exclusive with head_addr.

    Returns:
        ActorSystem instance

    Example:
        # Standalone mode
        await init()

        # Cluster mode (Gossip + seed)
        await init(addr="0.0.0.0:8001", seeds=["192.168.1.1:8000"])

        # Head node
        await init(addr="0.0.0.0:8000", is_head_node=True)

        # Worker node
        await init(addr="0.0.0.0:8001", head_addr="192.168.1.1:8000")
    """
    global _global_system

    if _global_system is not None:
        return _global_system

    if is_head_node and head_addr:
        raise ValueError("Cannot set both is_head_node and head_addr")

    # Build config
    if addr:
        config = SystemConfig.with_addr(addr)
    else:
        config = SystemConfig.standalone()

    if seeds:
        config = config.with_seeds(seeds)
    if is_head_node:
        config = config.with_head_node()
    elif head_addr:
        config = config.with_head_addr(head_addr)

    if passphrase:
        config = config.with_passphrase(passphrase)

    loop = asyncio.get_running_loop()
    _global_system = await ActorSystem.create(config, loop)
    # Automatically register PythonActorService for remote actor creation
    from .remote import PYTHON_ACTOR_SERVICE_NAME, PythonActorService

    service = PythonActorService(_global_system)
    await _global_system.spawn(service, name=PYTHON_ACTOR_SERVICE_NAME, public=True)
    return _global_system


async def shutdown() -> None:
    """Shutdown the global actor system"""
    global _global_system

    if _global_system is not None:
        await _global_system.shutdown()
        _global_system = None


def get_system() -> ActorSystem:
    """Get the global actor system (must call init() first)"""
    if _global_system is None:
        from pulsing.exceptions import PulsingRuntimeError

        raise PulsingRuntimeError(
            "Actor system not initialized. Call 'await init()' first."
        )
    return _global_system


def is_initialized() -> bool:
    """Check if the global actor system is initialized"""
    return _global_system is not None


# =============================================================================
# Timeout utilities for cancellation support
# =============================================================================

# Default timeout for ask operations (seconds)
DEFAULT_ASK_TIMEOUT = 30.0


async def ask_with_timeout(
    actor_ref: ActorRef,
    msg: Any,
    timeout: float = DEFAULT_ASK_TIMEOUT,
) -> Any:
    """Send a message and wait for response with timeout.

    This is a convenience wrapper around ActorRef.ask() that adds timeout support.
    When timeout occurs, the local task is cancelled. Note that this does NOT
    guarantee the remote handler will stop - it relies on HTTP/2 RST_STREAM
    propagation for stream cancellation.

    For handlers that may run long, implement idempotent operations and/or
    check for stream closure in streaming scenarios.

    Args:
        actor_ref: Target actor reference
        msg: Message to send (any Python object or Message)
        timeout: Timeout in seconds (default: 30.0)

    Returns:
        Response from the actor

    Raises:
        asyncio.TimeoutError: If timeout expires before response
        Exception: Any error from the actor

    Example:
        try:
            result = await ask_with_timeout(actor_ref, {"action": "compute"}, timeout=10.0)
        except asyncio.TimeoutError:
            print("Request timed out")
    """
    return await asyncio.wait_for(actor_ref.ask(msg), timeout=timeout)


async def tell_with_timeout(
    actor_ref: ActorRef,
    msg: Any,
    timeout: float = DEFAULT_ASK_TIMEOUT,
) -> None:
    """Send a fire-and-forget message with timeout.

    Args:
        actor_ref: Target actor reference
        msg: Message to send
        timeout: Timeout in seconds (default: 30.0)

    Raises:
        asyncio.TimeoutError: If timeout expires
    """
    await asyncio.wait_for(actor_ref.tell(msg), timeout=timeout)


from . import helpers
from .remote import (
    PYTHON_ACTOR_SERVICE_NAME,
    ActorClass,
    ActorProxy,
    PythonActorService,
    PythonActorServiceProxy,
    SystemActorProxy,
    as_any,
    get_metrics,
    get_node_info,
    get_python_actor_service,
    get_system_actor,
    health_check,
    list_actors,
    mount,
    unmount,
    ping,
    remote,
    resolve,
)

# Import exceptions for convenience
from pulsing.exceptions import (
    PulsingError,
    PulsingRuntimeError,
    PulsingActorError,
)

__all__ = [
    "init",
    "shutdown",
    "remote",
    "resolve",
    "mount",
    "unmount",
    "get_system",
    "get_system_actor",
    "is_initialized",
    "Actor",
    "Message",
    "StreamMessage",
    "SystemConfig",
    "ActorSystem",
    "ActorRef",
    "ActorId",
    "ActorProxy",
    "as_any",
    "SystemActorProxy",
    "PythonActorService",
    "PYTHON_ACTOR_SERVICE_NAME",
    "ZeroCopyDescriptor",
    "PulsingError",
    "PulsingRuntimeError",
    "PulsingActorError",
]


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
