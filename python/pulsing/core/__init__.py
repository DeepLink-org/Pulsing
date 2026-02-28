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

from pulsing._core import (
    ActorId,
    ActorRef,
    ActorSystem,
    NodeId,
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


from . import helpers
from .remote import (
    PYTHON_ACTOR_SERVICE_NAME,
    Actor,
    ActorClass,
    ActorProxy,
    PythonActorService,
    PythonActorServiceProxy,
    SystemActorProxy,
    get_python_actor_service,
    get_system_actor,
    remote,
    resolve,
)
from .helpers import mount, unmount

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
    "SystemActorProxy",
    "ZeroCopyDescriptor",
    "PulsingError",
    "PulsingRuntimeError",
    "PulsingActorError",
]
