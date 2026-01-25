"""
Pulsing - Distributed Actor Framework

Two API styles:

1. Actor System style (explicit system management):
    import pulsing as pul

    system = await pul.actor_system()

    @pul.remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = await Counter.spawn(name="counter")
    result = await counter.incr()

    await system.shutdown()

2. Ray-style async API (global system):
    import pulsing as pul

    await pul.init()

    @pul.remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = await Counter.spawn(name="counter")
    result = await counter.incr()

    await pul.shutdown()

3. Ray-compatible sync API (for migration):
    from pulsing.compat import ray

    ray.init()

    @ray.remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = Counter.remote(init=10)
    result = ray.get(counter.incr.remote())

    ray.shutdown()

Submodules:
- pulsing.actor: Native async API (recommended)
- pulsing.compat.ray: Ray-compatible sync API (for migration)
"""

import asyncio
from typing import Any

__version__ = "0.1.0"

# Import from pulsing.actor
from pulsing.actor import (
    # Global system functions
    init,
    shutdown,
    get_system,
    is_initialized,
    # Decorator
    remote,
    # Resolve function
    resolve,
    # Types
    Actor,
    ActorSystem as _ActorSystem,
    ActorRef,
    ActorId,
    ActorProxy,
    Message,
    StreamMessage,
    SystemConfig,
    # Service
    PythonActorService,
    PYTHON_ACTOR_SERVICE_NAME,
)

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
    """ActorSystem wrapper with queue API

    This wraps the Rust ActorSystem and adds Python-level extensions
    like the queue API.
    """

    def __init__(self, inner: _ActorSystem):
        self._inner = inner
        from pulsing.queue import QueueAPI

        self.queue = QueueAPI(inner)

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
    service = PythonActorService(inner)
    await inner.spawn(service, name=PYTHON_ACTOR_SERVICE_NAME, public=True)

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
    return await system.refer(actorid)


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
    # Types
    "Actor",
    "ActorSystem",
    "ActorRef",
    "ActorId",
    "ActorProxy",
    "Message",
    "StreamMessage",
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
