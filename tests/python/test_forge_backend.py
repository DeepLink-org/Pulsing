# SPDX-License-Identifier: Apache-2.0
"""Tests for ForgeBackend and Pulsing Actor integration helpers."""

from __future__ import annotations

import asyncio

import pytest

from pulsing.forge.tool_coverage import assert_forge_tool_coverage
from pulsing.forge.backend import (
    ForgeBackend,
    ForgeBackendMode,
    ForgeHostConfig,
    create_host_runtime,
)
from pulsing.forge.integrated import FORGE_HOST_TOOL_NAMES, FORGE_ISOLATED_TOOL_NAMES
from pulsing.forge.naming import shared_tool_worker_name


def test_shared_tool_worker_name() -> None:
    assert shared_tool_worker_name("abc123") == "craft/ws/abc123/_tools"


def test_forge_backend_modes_disjoint_tool_sets() -> None:
    assert not (FORGE_ISOLATED_TOOL_NAMES & FORGE_HOST_TOOL_NAMES)
    assert len(FORGE_ISOLATED_TOOL_NAMES) + len(FORGE_HOST_TOOL_NAMES) == 32


def test_create_host_runtime_local() -> None:
    rt = create_host_runtime(ForgeHostConfig(cwd=".", auto_approve=True))
    out = rt.call_tool("get_context_remaining", {})
    assert not out.is_error


def test_forge_tool_schema_coverage() -> None:
    assert_forge_tool_coverage()


def test_forge_backend_mode_enum() -> None:
    assert ForgeBackendMode.DEDICATED.value == "dedicated"
    assert ForgeBackendMode.SHARED.value == "shared"


class _FlakyWorker:
    """Minimal ForgeIsolatedWorker stand-in: fails once, then succeeds."""

    def __init__(self, mode: ForgeBackendMode) -> None:
        self.mode = mode
        self._lock = asyncio.Lock()
        self.calls = 0
        self.respawns = 0

    async def call_tool(self, name, arguments=None, *, event_sink=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("worker died")
        return {"content": "ok", "is_error": False}

    async def respawn(self, *, reason: str) -> None:
        self.respawns += 1


@pytest.mark.asyncio
async def test_forge_backend_call_tool_retries_dedicated_worker() -> None:
    """A failed isolated call must respawn (not crash with AttributeError/deadlock)."""
    worker = _FlakyWorker(ForgeBackendMode.DEDICATED)
    backend = ForgeBackend(
        host=create_host_runtime(ForgeHostConfig(cwd=".")), worker=worker
    )

    name = next(iter(FORGE_ISOLATED_TOOL_NAMES))
    result = await asyncio.wait_for(backend.call_tool(name, {}), timeout=5.0)

    assert not result.is_error
    assert worker.calls == 2
    assert worker.respawns == 1


@pytest.mark.asyncio
async def test_forge_backend_call_tool_retries_shared_worker() -> None:
    """Shared-mode recovery must only clear the local proxy (no respawn)."""
    worker = _FlakyWorker(ForgeBackendMode.SHARED)
    backend = ForgeBackend(
        host=create_host_runtime(ForgeHostConfig(cwd=".")), worker=worker
    )

    name = next(iter(FORGE_ISOLATED_TOOL_NAMES))
    result = await asyncio.wait_for(backend.call_tool(name, {}), timeout=5.0)

    assert not result.is_error
    assert worker.calls == 2
    assert worker.respawns == 0
