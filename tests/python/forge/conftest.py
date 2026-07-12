# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for Forge gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE
from pulsing.testing.forge_harness import local_runtime


@pytest.fixture
def forge_workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def local_forge(forge_workspace: Path):
    return local_runtime(forge_workspace)


@pytest.fixture
def hybrid_forge(forge_workspace: Path):
    from pulsing.forge.hybrid_runtime import HybridForgeRuntime

    if not RUST_FORGE_AVAILABLE:
        pytest.skip("requires maturin develop")
    return HybridForgeRuntime.create(
        cwd=str(forge_workspace),
        auto_approve=True,
    )
