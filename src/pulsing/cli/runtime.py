"""Common runtime creation utilities for backend workers."""

import asyncio
import signal

from hyperparameter import auto_param

from dynamo.runtime import DistributedRuntime


async def graceful_shutdown(runtime: DistributedRuntime):
    """Shutdown dynamo distributed runtime."""
    runtime.shutdown()


@auto_param("runtime")
def create_runtime(
    request_plane: str = "http",
    store_kv: str = "file",
) -> DistributedRuntime:
    """
    Create a DistributedRuntime instance.

    Args:
        request_plane: Request distribution method. Options: 'nats', 'http', 'tcp'. Default: 'http'
        store_kv: Key-value backend. Options: 'etcd', 'mem', 'file'. Default: 'file'

    Returns:
        DistributedRuntime instance
    """
    loop = asyncio.get_running_loop()
    return DistributedRuntime(loop, store_kv, request_plane)


def setup_signal_handlers(runtime: DistributedRuntime):
    """
    Setup signal handlers for graceful shutdown.

    Args:
        runtime: The DistributedRuntime instance to shutdown on signals
    """
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
