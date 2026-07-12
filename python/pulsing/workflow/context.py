# SPDX-License-Identifier: Apache-2.0
"""Minimal workflow API — plain Python orchestration on the Pulsing workspace."""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pulsing.workspace.journal import checkpoint, rollback
from pulsing.workspace.layout import (
    WorkspaceLayout,
    find_workspace_root,
    require_workspace_root,
)


class WorkflowContext:
    """Handle passed to user workflow functions.

    Bridges extension-mode Python to workspace journal and safe-mode agent.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        if root is not None:
            self._root = Path(root).resolve()
        elif env_root := os.environ.get("PULSING_WORKSPACE_ROOT", "").strip():
            self._root = Path(env_root).resolve()
        else:
            self._root = require_workspace_root()
        self._layout = WorkspaceLayout(self._root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def layout(self) -> WorkspaceLayout:
        return self._layout

    def info(self, message: str) -> None:
        print(f"[workflow] {message}", flush=True)

    def read_text(self, rel_path: str | Path, *, encoding: str = "utf-8") -> str:
        return (self._root / rel_path).read_text(encoding=encoding)

    def write_text(
        self,
        rel_path: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        path = self._root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    def checkpoint(self, message: str = "workflow", *, author: str = "workflow") -> str:
        manifest = checkpoint(self._layout, message=message, author=author)
        return str(manifest["id"])

    def rollback(self, revision: str | None = None) -> str:
        manifest = rollback(self._layout, revision_id=revision)
        return str(manifest["id"])

    def agent_sync(self, prompt: str) -> str:
        """Run safe-mode agent (one-shot).

        Inside ``pulsing run`` session, prefer the ``›`` prompt after workflow completes.
        """
        if os.environ.get("PULSING_WORKFLOW_SESSION") == "1":
            raise RuntimeError(
                "ctx.agent() during workflow run would nest sessions — "
                "orchestrate in Python, then use the › prompt for agent tasks after `pulsing run`",
            )
        binary = os.environ.get("PULSING_BINARY", "pulsing")
        result = subprocess.run(
            [binary, "agent", prompt],
            cwd=self._root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "agent failed").strip()
            raise RuntimeError(detail)
        return result.stdout.strip()

    async def agent(self, prompt: str) -> str:
        return await asyncio.to_thread(self.agent_sync, prompt)


def run(
    main: Callable[[WorkflowContext], Awaitable[None] | None],
    *,
    root: Path | str | None = None,
) -> None:
    """Execute a workflow entrypoint (sync or async)."""
    ctx = WorkflowContext(root=root)
    if inspect.iscoroutinefunction(main):
        _run_async(main(ctx))
        return
    result = main(ctx)
    if inspect.isawaitable(result):
        _run_async(result)


def _run_async(coro: Awaitable[None]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, coro).result()


def discover_workspace_root(start: Path | None = None) -> Path | None:
    """Return workspace root if present (non-exiting)."""
    return find_workspace_root(start)
