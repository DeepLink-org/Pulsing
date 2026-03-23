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
"""

import asyncio
import os

from pulsing._core import (
    ActorId,
    ActorRef,
    ActorSystem,
    NodeId,
    StreamReader,
    StreamWriter,
    SystemConfig,
    ZeroCopyDescriptor,
)
from .messaging import (
    Message,
    StreamMessage,
)  # internal: used by service.py / integrations

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
    ray_init: bool | None = None,
    ray_address: str | None = None,
) -> ActorSystem:
    """Initialize Pulsing actor system

    Args:
        addr: Bind address (e.g., "0.0.0.0:8000"). None for standalone mode.
        seeds: Seed nodes to join cluster (Gossip mode).
        passphrase: Enable TLS with this passphrase.
        head_addr: Address of head node (worker mode). Mutually exclusive with is_head_node.
        is_head_node: If True, this node runs as head. Mutually exclusive with head_addr.
        ray_init: Explicitly enable/disable Ray initialization. Falls back to PULSING_RAY_INIT env var.
        ray_address: Address passed to ray.init(address=...). None starts a new local cluster. Falls back to PULSING_RAY_ADDRESS env var.

    Returns:
        ActorSystem instance
    """
    global _global_system

    if _global_system is not None:
        return _global_system

    if is_head_node and head_addr:
        raise ValueError("Cannot set both is_head_node and head_addr")

    # Optionally initialize Ray
    should_ray = (
        ray_init
        if ray_init is not None
        else os.environ.get("PULSING_RAY_INIT", "").lower() in ("1", "true")
    )
    if should_ray:
        try:
            import ray

            if not ray.is_initialized():
                _ray_addr = (
                    ray_address
                    if ray_address is not None
                    else os.environ.get("PULSING_RAY_ADDRESS")
                )
                ray.init(address=_ray_addr)
        except ImportError:
            pass

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

    service = PythonActorService(_global_system)
    await _global_system.spawn(service, name=PYTHON_ACTOR_SERVICE_NAME, public=True)

    # Notify pulsing.subprocess of the loop so sync callers can submit coroutines
    try:
        from pulsing.subprocess.popen import _set_pulsing_loop

        _set_pulsing_loop(loop)
    except ImportError:
        pass

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


from . import helpers  # noqa: E402
from .helpers import mount, unmount  # noqa: E402
from .proxy import ActorProxy  # noqa: E402
from .remote import (  # noqa: E402
    Actor,
    ActorClass,
    remote,
    resolve,
)
from .service import (  # noqa: E402
    PYTHON_ACTOR_SERVICE_NAME,
    PythonActorService,
    PythonActorServiceProxy,
    SystemActorProxy,
    get_python_actor_service,
    get_system_actor,
)

from pulsing.exceptions import (  # noqa: E402
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
    "SystemConfig",
    "ActorSystem",
    "ActorRef",
    "ActorId",
    "ActorProxy",
    "SystemActorProxy",
    "PulsingError",
    "PulsingRuntimeError",
    "PulsingActorError",
]
