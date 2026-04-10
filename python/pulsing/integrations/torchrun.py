"""
pulsing.torchrun - Initialize Pulsing in torchrun / torch.distributed

For automatic cluster formation, use pulsing.bootstrap(ray=False, torchrun=True).
This module provides init_in_torchrun(), used by bootstrap.

Rank 0 starts Pulsing and broadcasts its listen address to other ranks via
torch.distributed.broadcast_object_list(); others join with seeds=[rank0_addr].

Requires torch.distributed.init_process_group() to be called first (e.g. by torchrun).

Usage:
    # Recommended: use bootstrap in your script (launched with torchrun)
    import pulsing as pul
    pul.bootstrap(ray=False, torchrun=True, wait_timeout=30)

    # Or call init_in_torchrun directly after init_process_group
    import torch.distributed as dist
    from pulsing.integrations.torchrun import init_in_torchrun
    dist.init_process_group(...)
    init_in_torchrun()  # rank 0 becomes seed, others join
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

try:
    import torch.distributed as dist
except ImportError:
    raise ImportError(
        "pulsing.integrations.torchrun requires PyTorch. Install with: pip install torch"
    )

from pulsing._async_bridge import submit_on_shared_loop

if TYPE_CHECKING:
    from pulsing.core import ActorSystem


async def _do_init(addr: str, seeds: list[str] | None = None):
    from pulsing.core import init

    return await init(addr=addr, seeds=seeds)


def _get_master_addr() -> str:
    return os.environ.get("MASTER_ADDR", "127.0.0.1")


def init_in_torchrun() -> ActorSystem:
    """Initialize Pulsing in current process and join cluster using torch.distributed.

    Rank 0 binds to 0.0.0.0:0, then broadcasts MASTER_ADDR:port to all ranks.
    Other ranks receive the seed address and init with seeds=[seed_addr].

    Must be called after torch.distributed.init_process_group() (e.g. under torchrun).
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed not initialized. Call torch.distributed.init_process_group() first (e.g. use torchrun)."
        )

    rank = dist.get_rank()
    master_addr = _get_master_addr()

    if rank == 0:
        # Rank 0: start Pulsing, get bound port, advertise MASTER_ADDR:port
        system = submit_on_shared_loop(_do_init("0.0.0.0:0"), timeout=60)
        bound = str(system.addr)
        # bound is e.g. "0.0.0.0:12345"; advertise as MASTER_ADDR:12345
        port = bound.split(":")[-1]
        seed_addr = f"{master_addr}:{port}"
        object_list = [seed_addr]
    else:
        object_list = [None]

    dist.broadcast_object_list(object_list, src=0)
    seed_addr = object_list[0]

    if rank == 0:
        return system

    # Non-rank0: join with seed
    return submit_on_shared_loop(_do_init("0.0.0.0:0", seeds=[seed_addr]), timeout=60)


async def async_init_in_torchrun() -> ActorSystem:
    """Initialize Pulsing under torch.distributed (async version)."""
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed not initialized. Call torch.distributed.init_process_group() first."
        )

    rank = dist.get_rank()
    master_addr = _get_master_addr()

    if rank == 0:
        system = await _do_init("0.0.0.0:0")
        bound = str(system.addr)
        port = bound.split(":")[-1]
        seed_addr = f"{master_addr}:{port}"
        object_list = [seed_addr]
    else:
        object_list = [None]

    dist.broadcast_object_list(object_list, src=0)
    seed_addr = object_list[0]

    if rank == 0:
        return system
    return await _do_init("0.0.0.0:0", seeds=[seed_addr])


__all__ = ["init_in_torchrun", "async_init_in_torchrun"]
