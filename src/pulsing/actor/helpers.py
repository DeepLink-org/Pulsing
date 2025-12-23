"""Actor helper functions - simplify actor creation and lifecycle management"""

import asyncio
import signal
from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from . import Actor, ActorSystem, SystemConfig, ActorRef


async def spawn_service_actor(
    actor: "Actor",
    name: str,
    addr: Optional[str] = None,
    seeds: Optional[List[str]] = None,
    public: bool = True,
) -> Tuple["ActorSystem", "ActorRef"]:
    """
    Create ActorSystem and spawn an actor
    
    Args:
        actor: Actor instance
        name: Actor name
        addr: Bind address (e.g. "0.0.0.0:8000")
        seeds: Seed node list
        public: Whether to register as public actor for discovery
    
    Returns:
        (ActorSystem, ActorRef) tuple
    """
    from . import ActorSystem, SystemConfig
    
    if addr:
        config = SystemConfig.with_addr(addr)
    else:
        config = SystemConfig.standalone()
    
    if seeds:
        config = config.with_seeds(seeds)
    
    system = await ActorSystem.create(config)
    actor_ref = await system.spawn(name, actor, public=public)
    
    return system, actor_ref


async def run_until_signal(system: "ActorSystem", actor_name: Optional[str] = None):
    """
    Run until shutdown signal (SIGTERM or SIGINT)
    
    Args:
        system: ActorSystem instance
        actor_name: Actor name for logging
    """
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        print(f"[{actor_name or 'Actor'}] Received shutdown signal")
        asyncio.create_task(shutdown_system())
    
    async def shutdown_system():
        try:
            if actor_name:
                await system.stop(actor_name)
        except Exception as e:
            print(f"[{actor_name or 'Actor'}] Stop error: {e}")
        
        try:
            await system.shutdown()
        except Exception as e:
            print(f"[{actor_name or 'Actor'}] Shutdown error: {e}")
        
        shutdown_event.set()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    await shutdown_event.wait()
    print(f"[{actor_name or 'Actor'}] Stopped")


async def spawn_and_run(
    actor: "Actor",
    name: str,
    addr: Optional[str] = None,
    seeds: Optional[List[str]] = None,
    public: bool = True,
):
    """
    Create ActorSystem, spawn actor, and run until signal
    
    Combines spawn_service_actor + run_until_signal.
    
    Args:
        actor: Actor instance
        name: Actor name
        addr: Bind address
        seeds: Seed node list
        public: Whether to register as public actor
    """
    system, actor_ref = await spawn_service_actor(actor, name, addr, seeds, public)
    print(f"[{name}] Started at {system.addr}")
    await run_until_signal(system, name)
