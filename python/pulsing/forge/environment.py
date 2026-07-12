# SPDX-License-Identifier: Apache-2.0
"""Agent execution environment — the primary Forge abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from pulsing.forge.hybrid_runtime import HybridForgeRuntime
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.rust_runtime import rust_forge_available
from pulsing.forge.session import LocalToolSession, NullToolSession, ToolSession

ForgeRuntime = Union[LocalToolRuntime, HybridForgeRuntime]


@dataclass
class ForgeEnvironment:
    """Where an agent runs tools: workspace root, sandbox, and host session hooks."""

    cwd: Path | str = "."
    sandbox_policy: str = "off"
    dangerously_disable_sandbox: bool = False
    session: ToolSession = field(default_factory=LocalToolSession)
    auto_approve: bool = False

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd).resolve()

    def runtime(self) -> ForgeRuntime:
        """Preferred runtime: Hybrid (Rust+Python) when built, else Python-only."""
        if rust_forge_available():
            return HybridForgeRuntime.create(
                cwd=str(self.cwd),
                sandbox_policy=self.sandbox_policy,
                dangerously_disable_sandbox=self.dangerously_disable_sandbox,
                session=self.session,
                auto_approve=self.auto_approve,
            )
        return LocalToolRuntime(
            cwd=str(self.cwd),
            sandbox_policy=self.sandbox_policy,
            dangerously_disable_sandbox=self.dangerously_disable_sandbox,
            session=self.session,
        )

    @classmethod
    def ephemeral(
        cls, cwd: Path | str = ".", *, auto_approve: bool = True
    ) -> ForgeEnvironment:
        """Environment with no host session side effects (shell/files only)."""
        return cls(cwd=cwd, session=NullToolSession(), auto_approve=auto_approve)
