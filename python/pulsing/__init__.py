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

from __future__ import annotations

import asyncio
from typing import Any

__version__ = "0.1.2"


def __getattr__(name: str) -> Any:
    """PEP 562: recover from incomplete package init left by circular imports.

    Must be defined *before* eager imports below so re-entrant ``pul.remote``
    during those imports can resolve via submodule load.
    """
    import importlib

    if name == "transfer_queue":
        value = importlib.import_module("pulsing.transfer_queue")
        globals()[name] = value
        return value
    # Prefer submodule import — works even while pulsing.core.__init__ is mid-flight.
    if name in {"remote", "resolve", "Actor", "ActorClass"}:
        mod = importlib.import_module("pulsing.core.remote")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    if name in {
        "init",
        "shutdown",
        "get_system",
        "is_initialized",
        "mount",
        "unmount",
        "ActorRef",
        "ActorId",
        "ActorProxy",
        "SystemConfig",
    }:
        mod = importlib.import_module("pulsing.core")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Bind critical APIs via submodule so circular imports during package init
# (e.g. agent/forge/subprocess pulling `import pulsing as pul`) still see @remote.
from pulsing.core.remote import Actor, ActorClass, remote, resolve

from pulsing.core import (
    # Global system functions
    init,
    shutdown,
    get_system,
    is_initialized,
    # Mount (attach existing object to Pulsing network)
    mount,
    unmount,
    # Types (Actor / remote / resolve already bound above)
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
from pulsing.core.isolated_bridge import IsolatedSpawnHandle

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

    async def spawn(
        self,
        actor: Any | None = None,
        *,
        new_process: bool = False,
        child_addr: str = "127.0.0.1:0",
        child_seed: str | None = None,
        child_passphrase: str | None = None,
        child_extra_env: dict[str, str] | None = None,
        name: str | None = None,
        public: bool = False,
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0,
    ) -> ActorRef | asyncio.subprocess.Process | IsolatedSpawnHandle:
        return await _spawn_dispatch(
            self._inner,
            actor,
            new_process=new_process,
            child_addr=child_addr,
            child_seed=child_seed,
            child_passphrase=child_passphrase,
            child_extra_env=child_extra_env,
            name=name,
            public=public,
            restart_policy=restart_policy,
            max_restarts=max_restarts,
            min_backoff=min_backoff,
            max_backoff=max_backoff,
        )

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


async def _spawn_dispatch(
    system: _ActorSystem,
    actor: Any | None,
    *,
    new_process: bool,
    child_addr: str,
    child_seed: str | None,
    child_passphrase: str | None,
    child_extra_env: dict[str, str] | None,
    name: str | None,
    public: bool,
    restart_policy: str,
    max_restarts: int,
    min_backoff: float,
    max_backoff: float,
) -> ActorRef | asyncio.subprocess.Process | IsolatedSpawnHandle:
    if new_process:
        if actor is not None:
            from pulsing.core.isolated_spawn import spawn_isolated_actor

            return await spawn_isolated_actor(
                system,
                actor,
                name=name,
                public=public,
                restart_policy=restart_policy,
                max_restarts=max_restarts,
                min_backoff=min_backoff,
                max_backoff=max_backoff,
            )
        from pulsing.cluster_spawn import _spawn_cluster_child_async

        return await _spawn_cluster_child_async(
            system,
            child_addr=child_addr,
            seed_addr=child_seed,
            passphrase=child_passphrase,
            extra_env=child_extra_env,
        )
    if actor is None:
        raise TypeError("actor is required when new_process=False")
    return await system.spawn(
        actor,
        name=name,
        public=public,
        restart_policy=restart_policy,
        max_restarts=max_restarts,
        min_backoff=min_backoff,
        max_backoff=max_backoff,
    )


async def spawn(
    actor: Any | None = None,
    *,
    new_process: bool = False,
    child_addr: str = "127.0.0.1:0",
    child_seed: str | None = None,
    child_passphrase: str | None = None,
    child_extra_env: dict[str, str] | None = None,
    name: str | None = None,
    public: bool = False,
    restart_policy: str = "never",
    max_restarts: int = 3,
    min_backoff: float = 0.1,
    max_backoff: float = 30.0,
) -> ActorRef | asyncio.subprocess.Process | IsolatedSpawnHandle:
    """Spawn an actor on the global system, or start a cluster child OS process.

    Args:
        actor: Actor instance. With ``new_process=False`` it is required. With
            ``new_process=True``, pass ``None`` to start a full extra cluster member
            (``python -m pulsing.spawn_node``), or pass an actor instance to run that
            actor in an isolated child process connected via ``Connect`` while the
            cluster only sees the parent-side bridge actor.
        new_process: If True and ``actor is None``, start ``python -m pulsing.spawn_node``.
            If True and ``actor`` is set, start ``python -m pulsing.isolated_worker`` (MVP:
            ``restart_policy`` must be ``\"never\"``).
        child_addr: Bind address for the child node (default ``127.0.0.1:0``).
        child_seed: Override seed address for the child; default normalizes ``0.0.0.0`` to loopback.
        child_passphrase: TLS passphrase for the child (must match parent if TLS is enabled).
        child_extra_env: Extra environment variables for the child process.
        name, public, restart_policy, max_restarts, min_backoff, max_backoff: forwarded to the
            Rust actor spawn when ``new_process=False``.

    Returns:
        :class:`ActorRef` for a normal spawn, :class:`asyncio.subprocess.Process` when
        ``new_process=True`` and ``actor is None``, or :class:`IsolatedSpawnHandle`
        when ``new_process=True`` with an actor (``ref`` + ``process``).

    Example:
        import pulsing as pul

        await pul.init(addr="127.0.0.1:8000")

        class MyActor:
            async def receive(self, msg):
                return f"Got: {msg}"

        ref = await pul.spawn(MyActor(), name="my_actor")

        proc = await pul.spawn(None, new_process=True)
        proc.terminate()
    """
    system = get_system()
    return await _spawn_dispatch(
        system,
        actor,
        new_process=new_process,
        child_addr=child_addr,
        child_seed=child_seed,
        child_passphrase=child_passphrase,
        child_extra_env=child_extra_env,
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
    # Transfer queue
    "transfer_queue",
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
    "IsolatedSpawnHandle",
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
