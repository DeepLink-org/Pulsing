# SPDX-License-Identifier: Apache-2.0
"""Connect to workspace cluster (seed from ``.pulsing/node.json``)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

import pulsing as pul

from pulsing.agent.workspace.config import WorkspaceConfig


def require_seed(cfg: WorkspaceConfig) -> str:
    seed = cfg.seed_addr()
    if not seed:
        raise SystemExit(
            f"world asleep — run `pulsing agent wake` in {cfg.root}",
        )
    return seed


@asynccontextmanager
async def workspace_session(
    cfg: WorkspaceConfig,
    *,
    bind_addr: str | None = None,
) -> AsyncIterator[Any]:
    """Join workspace gossip; ``bind_addr`` set only for ``pulsing agent wake`` (local node)."""
    seeds: list[str] | None = None
    if bind_addr is None:
        seeds = [require_seed(cfg)]
    try:
        system = await pul.init(
            addr=bind_addr,
            seeds=seeds,
            passphrase=os.environ.get("PULSING_PASSPHRASE"),
        )
        yield system
    finally:
        await pul.shutdown()
