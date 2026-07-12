# SPDX-License-Identifier: Apache-2.0
"""Optional workspace-level shared isolated tool worker."""

from __future__ import annotations

from typing import Any

from pulsing.agent.workspace.config import WorkspaceConfig
from pulsing.forge.backend import spawn_shared_tool_worker as _spawn_shared
from pulsing.forge.backend import resolve_shared_tool_worker as _resolve_shared

SHARED_WORKER_SHORT = "_tools"


async def spawn_shared_tool_worker(cfg: WorkspaceConfig) -> Any:
    return await _spawn_shared(
        workspace_id=cfg.cluster_id,
        cwd=cfg.root,
        sandbox_policy=cfg.sandbox,
    )


async def resolve_shared_tool_worker(
    workspace_id: str, *, timeout: float = 120.0
) -> Any:
    return await _resolve_shared(workspace_id, timeout=timeout)
