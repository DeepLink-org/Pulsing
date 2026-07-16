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
import sys

from pulsing._async_bridge import (
    clear_pulsing_loop,
    set_pulsing_loop,
    _is_rustpython,
    _start_shared_loop,
)
from pulsing._core import (
    ActorId,
    ActorRef,
    ActorSystem,
    NodeId,
    StreamReader,
    StreamWriter,
    SystemConfig,
    ZeroCopyDescriptor,
    init_distributed_tracing,
    shutdown_distributed_tracing,
)
from .messaging import (
    Message,
    StreamMessage,
)  # internal: used by service.py / integrations

_native_core = sys.modules["pulsing._core"]
_HAS_NATIVE_TENSOR_TRANSPORT = all(
    hasattr(_native_core, name)
    for name in ("TensorMessage", "tensor_transport_stats")
)
if _HAS_NATIVE_TENSOR_TRANSPORT:
    TensorMessage = _native_core.TensorMessage
    tensor_transport_stats = _native_core.tensor_transport_stats

# =============================================================================
# Global system for simple API
# =============================================================================

_global_system: ActorSystem = None


def _init_rust_tracing_from_env() -> None:
    """Enable Rust `tracing` + OpenTelemetry layer when ``PULSING_TRACING`` is truthy."""
    if os.environ.get("PULSING_TRACING", "").lower() not in ("1", "true", "yes"):
        return
    console = os.environ.get("PULSING_TRACING_CONSOLE", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    service = os.environ.get("PULSING_SERVICE_NAME") or None
    init_distributed_tracing(
        service_name=service,
        console_output=console,
    )


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

    Tracing:
        Set ``PULSING_TRACING=1`` to install the Rust subscriber (console + OTLP-ready layer).
        Optional: ``PULSING_OTLP_ENDPOINT``, ``PULSING_SERVICE_NAME``, ``PULSING_TRACING_CONSOLE``,
        ``PULSING_SPAN_HISTORY_CAPACITY`` (max retained completed spans, default 8192).

    If the ``probing`` package is installed, ``PULSING_PROBING_AUTO_TRACING`` (default ``1``) may
    install silent Rust tracing so ``pulsing.spans`` gets populated.

    Returns:
        ActorSystem instance
    """
    global _global_system

    if _global_system is not None:
        return _global_system

    _init_rust_tracing_from_env()

    try:
        import probing  # noqa: F401

        from pulsing.integrations.probing import ensure_tracing_for_span_capture

        ensure_tracing_for_span_capture()
    except ImportError:
        pass

    if is_head_node and head_addr:
        raise ValueError("Cannot set both is_head_node and head_addr")

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
    set_pulsing_loop(loop)

    try:
        from pulsing.integrations.probing import start_probing_integration

        start_probing_integration()
    except ImportError:
        pass

    return _global_system


async def shutdown() -> None:
    """Shutdown the global actor system"""
    global _global_system

    try:
        from pulsing.integrations.probing import stop_probing_integration

        stop_probing_integration()
    except ImportError:
        pass

    if _global_system is not None:
        await _global_system.shutdown()
        _global_system = None
    clear_pulsing_loop()

    try:
        from pulsing._runtime import clear_module_ownership

        clear_module_ownership()
    except ImportError:
        pass

    shutdown_distributed_tracing()


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


def _cli_attach_from_native(system: ActorSystem) -> None:
    """Attach Rust-started ActorSystem (pulsing-cli / RustPython Path B)."""
    global _global_system
    if _global_system is not None:
        return
    _global_system = system
    if _is_rustpython():
        # RustPython threading is unreliable; avoid background loop at attach.
        return
    loop = _start_shared_loop()
    set_pulsing_loop(loop)


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
    "init_distributed_tracing",
    "shutdown_distributed_tracing",
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

if _HAS_NATIVE_TENSOR_TRANSPORT:
    __all__.extend(["TensorMessage", "tensor_transport_stats"])
