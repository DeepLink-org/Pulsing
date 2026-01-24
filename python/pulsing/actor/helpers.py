"""Actor helper functions - simplify actor creation and lifecycle management"""

import asyncio
import signal
import sys
from typing import TYPE_CHECKING

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
