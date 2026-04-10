"""Actor helper functions - lifecycle management and sync/async bridge."""

import asyncio
import signal
import sys
from typing import TYPE_CHECKING, Any

from pulsing._async_bridge import (
    get_loop,
    get_shared_loop,
    run_sync as _bridge_run_sync,
)

if TYPE_CHECKING:
    from . import Actor, ActorSystem


async def run_until_signal(actor_name: str | None = None) -> None:
    """
    Run until shutdown signal (SIGTERM or SIGINT)

    Handles graceful shutdown on first signal, force quits on second signal.
    Uses the global system via shutdown() to ensure proper cleanup.

    Args:
        actor_name: Optional actor name for logging
    """
    from . import get_system, shutdown

    shutdown_event = asyncio.Event()
    shutting_down = False

    def signal_handler():
        nonlocal shutting_down
        if shutting_down:
            print(f"[{actor_name or 'Actor'}] Force quit!")
            sys.exit(1)

        shutting_down = True
        print(f"[{actor_name or 'Actor'}] Received shutdown signal")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Wait for signal
    await shutdown_event.wait()

    # Perform graceful shutdown
    try:
        system = get_system()
        if actor_name:
            await system.stop(actor_name)
    except Exception as e:
        print(f"[{actor_name or 'Actor'}] Stop error: {e}")

    # Use module-level shutdown() to properly clear global state
    try:
        await shutdown()
    except Exception as e:
        print(f"[{actor_name or 'Actor'}] Shutdown error: {e}")

    print(f"[{actor_name or 'Actor'}] Stopped")


async def spawn_and_run(
    actor: "Actor",
    name: str,
    addr: str | None = None,
    seeds: list[str] | None = None,
    public: bool = True,
) -> None:
    """
    Create ActorSystem via init(), spawn actor, and run until signal

    This function uses init() to ensure the global system is set,
    making get_system() available inside actor on_start()/receive().

    Args:
        actor: Actor instance
        name: Actor name
        addr: Bind address (e.g. "0.0.0.0:8000")
        seeds: List of seed node addresses for cluster discovery
        public: Whether to register as public named actor
    """
    from . import get_system, init

    # Use init() to set global system (makes get_system() work inside actors)
    system = await init(addr=addr, seeds=seeds)
    await system.spawn(actor, name=name, public=public)

    print(f"[{name}] Started at {system.addr}")
    await run_until_signal(name)


# ---------------------------------------------------------------------------
# Sync/async bridge — single implementation used by mount/unmount
# ---------------------------------------------------------------------------


def run_sync(coro) -> Any:
    """Execute a coroutine synchronously on the Pulsing shared background event loop.

    Handles three environments:
    - Shared Pulsing background loop: submits to the singleton background loop
    - Standalone (no running loop): uses ``asyncio.run``
    - Inside a running loop (e.g. a Jupyter cell): runs in a thread-pool worker

    Raises:
        TimeoutError: if the coroutine does not complete within 30 s.
    """
    # Keep a concrete, patchable loop lookup here for backwards compatibility with
    # older tests and callers that monkeypatch ``get_shared_loop`` directly.
    dispatch_loop = get_loop() or get_shared_loop()
    return _bridge_run_sync(
        coro,
        loop=dispatch_loop,
        timeout=30,
        same_loop="worker",
        missing_loop="run",
    )


# ---------------------------------------------------------------------------
# mount / unmount — sync API to expose Python objects as Pulsing actors
# ---------------------------------------------------------------------------


def _auto_init_pulsing() -> None:
    """Auto-detect environment and initialize Pulsing."""
    try:
        import ray

        if ray.is_initialized():
            from pulsing.integrations.ray import init_in_ray

            init_in_ray()
            return
    except ImportError:
        pass

    raise RuntimeError(
        "Pulsing not initialized. Please call await pul.init() or run in Ray environment."
    )


def mount(instance: Any, *, name: str, public: bool = True) -> None:
    """Mount an existing Python object to the Pulsing communication network.

    Synchronous interface, can be called in ``__init__``. Automatically:
      1. Initialize Pulsing (if not already, auto-detects Ray environment)
      2. Wrap instance as a Pulsing actor
      3. Register to Pulsing network — other nodes can discover via ``pul.resolve(name)``

    Args:
        instance: Object to mount (any Python instance)
        name: Pulsing name, other nodes resolve via this name
        public: Whether discoverable by other cluster nodes (default True)

    Example::

        @ray.remote
        class Counter:
            def __init__(self, name, peers):
                self.name = name
                pul.mount(self, name=name)

            async def greet(self, msg):
                return f"Hello from {self.name}: {msg}"
    """
    from . import _global_system

    if _global_system is None:
        _auto_init_pulsing()

    from . import _global_system as system

    if system is None:
        raise RuntimeError(
            "Pulsing initialization failed. Please call pul.init() or run in Ray environment."
        )

    from .remote import _WrappedActor, _register_actor_metadata

    actor_name = name if "/" in name else f"actors/{name}"
    wrapped = _WrappedActor(instance)

    async def _do_mount():
        return await system.spawn(wrapped, name=actor_name, public=public)

    actor_ref = run_sync(_do_mount())
    wrapped._inject_delayed(actor_ref)
    _register_actor_metadata(actor_name, type(instance))


def unmount(name: str) -> None:
    """Unmount a previously mounted actor from the Pulsing network.

    Args:
        name: Name used during mounting
    """
    from . import _global_system

    if _global_system is None:
        return

    actor_name = name if "/" in name else f"actors/{name}"

    async def _do_unmount():
        await _global_system.stop(actor_name)

    run_sync(_do_unmount())
