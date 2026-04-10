"""
pulsing.ray - Initialize Pulsing in Ray cluster

For automatic cluster formation, use pulsing.bootstrap(ray=True, torchrun=False) (or
pul.bootstrap() to try both Ray and torchrun). This module provides init_in_ray(),
used by bootstrap and as worker_process_setup_hook so each Ray worker initializes Pulsing.

Uses Ray's internal KV store to coordinate seed node discovery.

Recommended usage:
    import ray
    from pulsing.integrations.ray import init_in_ray
    import pulsing as pul

    ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})
    pul.bootstrap(ray=True, torchrun=False, wait_timeout=30)  # driver
"""

try:
    import ray
    from ray.experimental.internal_kv import (
        _internal_kv_get,
        _internal_kv_put,
        _internal_kv_del,
    )
except ImportError:
    raise ImportError(
        "pulsing.integrations.ray requires Ray. Install with: pip install 'ray[default]'"
    )

from pulsing._async_bridge import submit_on_shared_loop

_SEED_KEY = "pulsing:seed_addr"


def _get_node_ip():
    """Get current Ray node IP"""
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    for node in ray.nodes():
        if node["NodeID"] == node_id and node["Alive"]:
            return node["NodeManagerAddress"]
    raise RuntimeError("Cannot get current Ray node IP")


async def _do_init(addr, seeds=None):
    from pulsing.core import init

    return await init(addr=addr, seeds=seeds)


async def _do_shutdown():
    from pulsing.core import shutdown

    await shutdown()


def _get_seed():
    """Get seed address from Ray KV store"""
    data = _internal_kv_get(_SEED_KEY)
    return data.decode() if data else None


def _try_set_seed(addr):
    """Atomically write seed address, returns True if write succeeded (I am seed).

    _internal_kv_put(overwrite=False) return value semantics:
        False = key doesn't exist, written (success)
        True  = key already exists, not overwritten (failure)
    """
    already_exists = _internal_kv_put(_SEED_KEY, addr.encode(), overwrite=False)
    return not already_exists


def init_in_ray():
    """Initialize Pulsing in current process and join cluster.

    Can be called directly or used as Ray worker_process_setup_hook:

        ray.init(runtime_env={"worker_process_setup_hook": init_in_ray})
        init_in_ray()  # driver also needs this
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized, please call ray.init() first")

    node_ip = _get_node_ip()

    # Seed exists -> join directly
    seed_addr = _get_seed()
    if seed_addr is not None:
        return submit_on_shared_loop(
            _do_init(f"{node_ip}:0", seeds=[seed_addr]), timeout=30
        )

    # Start as potential seed
    system = submit_on_shared_loop(_do_init(f"{node_ip}:0"), timeout=30)
    my_addr = str(system.addr)

    if _try_set_seed(my_addr):
        return system  # Write succeeded, I am seed

    # Race lost (rare), re-join with actual seed
    submit_on_shared_loop(_do_shutdown(), timeout=30)
    return submit_on_shared_loop(
        _do_init(f"{node_ip}:0", seeds=[_get_seed()]), timeout=30
    )


async def async_init_in_ray():
    """Initialize Pulsing in current process and join cluster (async version).

    Suitable for async Ray actors.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized, please call ray.init() first")

    node_ip = _get_node_ip()

    seed_addr = _get_seed()
    if seed_addr is not None:
        return await _do_init(f"{node_ip}:0", seeds=[seed_addr])

    system = await _do_init(f"{node_ip}:0")
    my_addr = str(system.addr)

    if _try_set_seed(my_addr):
        return system

    await _do_shutdown()
    return await _do_init(f"{node_ip}:0", seeds=[_get_seed()])


def cleanup():
    """Clean up Pulsing state in Ray KV store"""
    _internal_kv_del(_SEED_KEY)


__all__ = ["init_in_ray", "async_init_in_ray", "cleanup", "_get_seed"]
