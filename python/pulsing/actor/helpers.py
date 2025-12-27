"""Actor helper functions - simplify actor creation and lifecycle management"""

import asyncio
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Actor, ActorSystem


async def run_until_signal(
    system: "ActorSystem", actor_name: str | None = None
) -> None:
    """
    Run until shutdown signal (SIGTERM or SIGINT)

    Handles graceful shutdown on first signal, force quits on second signal.

    Args:
        system: ActorSystem instance
        actor_name: Optional actor name for logging
    """
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
        if actor_name:
            await system.stop(actor_name)
    except Exception as e:
        print(f"[{actor_name or 'Actor'}] Stop error: {e}")

    try:
        await system.shutdown()
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
    Create ActorSystem, spawn actor, and run until signal

    Args:
        actor: Actor instance
        name: Actor name
        addr: Bind address (e.g. "0.0.0.0:8000")
        seeds: List of seed node addresses for cluster discovery
        public: Whether to register as public named actor
    """
    from . import SystemConfig, create_actor_system

    config = SystemConfig.with_addr(addr) if addr else SystemConfig.standalone()
    if seeds:
        config = config.with_seeds(seeds)

    system = await create_actor_system(config)
    await system.spawn(name, actor, public=public)

    print(f"[{name}] Started at {system.addr}")
    await run_until_signal(system, name)
