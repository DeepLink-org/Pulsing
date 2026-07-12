"""Internal helpers for ``spawn(..., new_process=True)`` (child OS process + seed env)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from typing import Any


def normalize_seed_for_local_child(parent_addr: str) -> str:
    """Use loopback dial address when parent binds ``0.0.0.0`` (same machine)."""
    if parent_addr.startswith("0.0.0.0:"):
        port = parent_addr.removeprefix("0.0.0.0:")
        return f"127.0.0.1:{port}"
    return parent_addr


def build_cluster_child_env(
    *,
    child_addr: str,
    seed_addrs: list[str],
    passphrase: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {
        "PULSING_NODE_ADDR": child_addr,
        "PULSING_SEEDS": ",".join(seed_addrs),
    }
    if passphrase:
        env["PULSING_PASSPHRASE"] = passphrase
    if extra:
        env.update(extra)
    return env


def _spawn_cluster_child_sync(
    system: Any,
    *,
    child_addr: str = "127.0.0.1:0",
    seed_addr: str | None = None,
    passphrase: str | None = None,
    extra_env: dict[str, str] | None = None,
    **popen_kwargs: Any,
) -> subprocess.Popen:
    parent_addr = system.addr
    seed = (
        seed_addr
        if seed_addr is not None
        else normalize_seed_for_local_child(parent_addr)
    )
    env = os.environ.copy()
    env.update(
        build_cluster_child_env(
            child_addr=child_addr,
            seed_addrs=[seed],
            passphrase=passphrase,
            extra=extra_env,
        )
    )
    return subprocess.Popen(
        [sys.executable, "-m", "pulsing.spawn_node"],
        env=env,
        **popen_kwargs,
    )


async def _spawn_cluster_child_async(
    system: Any,
    *,
    child_addr: str = "127.0.0.1:0",
    seed_addr: str | None = None,
    passphrase: str | None = None,
    extra_env: dict[str, str] | None = None,
    **kwargs: Any,
) -> asyncio.subprocess.Process:
    parent_addr = system.addr
    seed = (
        seed_addr
        if seed_addr is not None
        else normalize_seed_for_local_child(parent_addr)
    )
    env = os.environ.copy()
    env.update(
        build_cluster_child_env(
            child_addr=child_addr,
            seed_addrs=[seed],
            passphrase=passphrase,
            extra=extra_env,
        )
    )
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pulsing.spawn_node",
        env=env,
        **kwargs,
    )
