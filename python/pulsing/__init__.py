"""
Pulsing - Distributed Actor Framework

Usage:
    import pulsing as pul

    await pul.init()

    @pul.remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = await Counter.spawn(name="counter")
    result = await counter.incr()

    await pul.shutdown()
"""

import asyncio
from typing import Any

__version__ = "0.1.2"

# Import from pulsing.core
from pulsing.core import (
    # Global system functions
    init,
    shutdown,
    get_system,
    is_initialized,
    # Decorator
    remote,
    # Resolve function
    resolve,
    # Mount (attach existing object to Pulsing network)
    mount,
    unmount,
    # Types
    Actor,
    ActorSystem as _ActorSystem,
    ActorRef,
    ActorId,
    ActorProxy,
    SystemConfig,
    # Service (internal, used by actor_system())
    PythonActorService as _PythonActorService,
    PYTHON_ACTOR_SERVICE_NAME as _PYTHON_ACTOR_SERVICE_NAME,
)


# Ray integration (lazy import — only available in Ray environment)
def init_inside_ray():
    """Initialize Pulsing in Ray worker and join cluster (async version).

    Usage::

        await pul.init_inside_ray()
    """
    from pulsing.integrations.ray import async_init_in_ray

    return async_init_in_ray()


def cleanup_ray():
    """Clean up Pulsing state in Ray KV store"""
    from pulsing.integrations.ray import cleanup

    return cleanup()


# torchrun / torch.distributed integration (lazy import)
def init_inside_torchrun():
    """Initialize Pulsing in current process and join cluster via torch.distributed.

    Rank 0 becomes the seed; others join with seeds=[rank0_addr]. Call after
    torch.distributed.init_process_group() (e.g. when launched with torchrun).

    Usage::

        import torch.distributed as dist
        dist.init_process_group(...)
        system = pul.init_inside_torchrun()
    """
    from pulsing.integrations.torchrun import init_in_torchrun

    return init_in_torchrun()


# Bootstrap: single API — pulsing.bootstrap(ray=..., torchrun=..., on_ready=..., wait_timeout=...)
from pulsing.bootstrap import bootstrap, stop as bootstrap_stop  # noqa: E402

bootstrap.stop = bootstrap_stop

# Import exceptions
from pulsing.exceptions import (
    PulsingError,
    PulsingRuntimeError,
    PulsingActorError,
    PulsingBusinessError,
    PulsingSystemError,
    PulsingTimeoutError,
    PulsingUnsupportedError,
)


class ActorSystem:
    """ActorSystem wrapper with queue/topic API

    This wraps the Rust ActorSystem and adds Python-level extensions
    like queue and topic APIs.
    """

    def __init__(self, inner: _ActorSystem):
        self._inner = inner
        from pulsing.streaming import QueueAPI, TopicAPI

        self.queue = QueueAPI(inner)
        self.topic = TopicAPI(inner)

    async def refer(self, actorid: ActorId | str) -> ActorRef:
        """Get actor reference by ID

        Args:
            actorid: Actor ID (ActorId instance or string in format "node_id:local_id")

        Returns:
            ActorRef to the actor
        """
        if isinstance(actorid, str):
            actorid = ActorId.from_str(actorid)
        return await self._inner.refer(actorid)

    def __getattr__(self, name):
        # Delegate all other attributes to the inner ActorSystem
        return getattr(self._inner, name)

    def __repr__(self):
        return f"ActorSystem(node_id={self._inner.node_id}, addr={self._inner.addr})"


async def actor_system(
    addr: str | None = None,
    *,
    seeds: list[str] | None = None,
    passphrase: str | None = None,
) -> ActorSystem:
    """Create a new ActorSystem (does not set global system)

    This is the Actor System style API for explicit system management.
    Use this when you need multiple systems or want explicit control.

    Args:
        addr: Bind address (e.g., "0.0.0.0:8000"). None for standalone mode.
        seeds: Seed nodes to join cluster
        passphrase: Enable TLS with this passphrase

    Returns:
        ActorSystem instance with .queue API

    Example:
        import pulsing as pul

        # Standalone mode
        system = await pul.actor_system()

        # Cluster mode
        system = await pul.actor_system(addr="0.0.0.0:8000")

        # Join existing cluster
        system = await pul.actor_system(
            addr="0.0.0.0:8001",
            seeds=["192.168.1.1:8000"]
        )

        # With TLS
        system = await pul.actor_system(
            addr="0.0.0.0:8000",
            passphrase="my-secret"
        )

        # Queue API
        writer = await system.queue.write("my_topic")
        reader = await system.queue.read("my_topic")
    """
    # Build config
    if addr:
        config = SystemConfig.with_addr(addr)
    else:
        config = SystemConfig.standalone()

    if seeds:
        config = config.with_seeds(seeds)

    if passphrase:
        config = config.with_passphrase(passphrase)

    loop = asyncio.get_running_loop()
    inner = await _ActorSystem.create(config, loop)

    # Wrap with Python ActorSystem
    system = ActorSystem(inner)

    # Automatically register PythonActorService (for remote actor creation)
    service = _PythonActorService(inner)
    await inner.spawn(service, name=_PYTHON_ACTOR_SERVICE_NAME, public=True)

    return system


async def spawn(
    actor: Any,
    *,
    name: str | None = None,
    public: bool = False,
    restart_policy: str = "never",
    max_restarts: int = 3,
    min_backoff: float = 0.1,
    max_backoff: float = 30.0,
) -> ActorRef:
    """Spawn an actor using the global system

    Args:
        actor: Actor instance
        name: Actor name (auto-generated if None)
        public: Whether to register as public named actor
        restart_policy: Restart policy ("never", "always", "on-failure")
        max_restarts: Maximum restart attempts
        min_backoff: Minimum backoff seconds
        max_backoff: Maximum backoff seconds

    Returns:
        ActorRef to the spawned actor

    Example:
        import pulsing as pul

        await pul.init()

        class MyActor:
            async def receive(self, msg):
                return f"Got: {msg}"

        ref = await pul.spawn(MyActor(), name="my_actor")
    """
    system = get_system()
    return await system.spawn(
        actor,
        name=name,
        public=public,
        restart_policy=restart_policy,
        max_restarts=max_restarts,
        min_backoff=min_backoff,
        max_backoff=max_backoff,
    )


async def refer(actorid: ActorId | str) -> ActorRef:
    """Get actor reference by ID using global system

    Args:
        actorid: Actor ID (ActorId instance or string)

    Returns:
        ActorRef to the actor
    """
    system = get_system()
    if isinstance(actorid, str):
        # Parse string to ActorId
        actorid = ActorId.from_str(actorid)
    if isinstance(actorid, int):
        actorid = ActorId(actorid)
    return await system.refer(actorid)


class _GlobalQueueAPI:
    """Lazy proxy for pul.queue that uses the global system."""

    async def write(self, topic, **kwargs):
        from pulsing.streaming import write_queue

        return await write_queue(get_system(), topic, **kwargs)

    async def read(self, topic, **kwargs):
        from pulsing.streaming import read_queue

        return await read_queue(get_system(), topic, **kwargs)


class _GlobalTopicAPI:
    """Lazy proxy for pul.topic that uses the global system."""

    async def write(self, topic, **kwargs):
        from pulsing.streaming import write_topic

        return await write_topic(get_system(), topic, **kwargs)

    async def read(self, topic, **kwargs):
        from pulsing.streaming import read_topic

        return await read_topic(get_system(), topic, **kwargs)


queue = _GlobalQueueAPI()
topic = _GlobalTopicAPI()


# Export all public APIs
__all__ = [
    # Version
    "__version__",
    # Actor System style API
    "actor_system",
    # Ray-style async API (global system)
    "init",
    "shutdown",
    "spawn",
    "refer",
    "resolve",
    "get_system",
    "is_initialized",
    # Decorator
    "remote",
    # Mount (attach existing object to Pulsing network)
    "mount",
    "unmount",
    # Queue & Topic (global entry points)
    "queue",
    "topic",
    # Ray integration
    "init_inside_ray",
    "cleanup_ray",
    # torchrun integration
    "init_inside_torchrun",
    # Bootstrap (auto cluster in background, wait_ready() for callers)
    "bootstrap",
    # Types
    "Actor",
    "ActorSystem",
    "ActorRef",
    "ActorId",
    "ActorProxy",
    # Exceptions
    "PulsingError",
    "PulsingRuntimeError",
    "PulsingActorError",
    # Business-level exceptions (automatically converted to ActorError)
    "PulsingBusinessError",
    "PulsingSystemError",
    "PulsingTimeoutError",
    "PulsingUnsupportedError",
]
