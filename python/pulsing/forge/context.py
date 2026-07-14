# SPDX-License-Identifier: Apache-2.0
"""Per tool-call context (cwd, sandbox, session)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsing.forge.discovery.catalog import ToolCatalog
from pulsing.forge.session import LocalToolSession, NullToolSession, ToolSession


def _new_exec() -> Any:
    from pulsing.forge.unified_exec import UnifiedExecManager

    return UnifiedExecManager()


def _new_catalog() -> ToolCatalog:
    catalog = ToolCatalog()
    catalog.load_codex_plugins()
    return catalog


def _new_code_mode() -> Any:
    from pulsing.forge.code_mode.service import CodeModeService

    return CodeModeService()


def _new_memories() -> Any:
    from pulsing.forge.extension.memories.local_backend import LocalMemoriesStore

    return LocalMemoriesStore()


def resolve_within_cwd(cwd: Path, path: str) -> Path:
    """Resolve `path` against `cwd`, rejecting targets outside the workspace."""
    joined = Path(path) if Path(path).is_absolute() else cwd / path
    target = joined.resolve()
    root = cwd.resolve()
    if target == root or root in target.parents:
        return target
    raise ValueError(
        f"refusing to write outside working directory: {target} (cwd: {root})"
    )


@dataclass
class ToolCallContext:
    cwd: Path
    sandbox_policy: str = "off"
    dangerously_disable_sandbox: bool = False
    session: ToolSession | None = None
    exec: Any = field(default_factory=_new_exec)
    tool_catalog: ToolCatalog = field(default_factory=_new_catalog)
    code_mode: Any = field(default_factory=_new_code_mode)
    memories: Any = field(default_factory=_new_memories)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd).resolve()
        if self.session is None:
            self.session = NullToolSession()

    @property
    def session_nonnull(self) -> ToolSession:
        assert self.session is not None
        return self.session
