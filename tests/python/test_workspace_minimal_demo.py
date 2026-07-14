# SPDX-License-Identifier: Apache-2.0
"""Smoke test for workspace_minimal_demo (offline demo LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.workspace.minimal_demo import run_workspace_minimal_demo


@pytest.mark.asyncio
async def test_workspace_minimal_demo_offline(tmp_path: Path) -> None:
    out = await run_workspace_minimal_demo(
        tmp_path,
        message="list project files with Glob",
        provider="demo",
    )
    text = str(out.get("assistant_text") or out.get("error") or "")
    assert text
    assert out.get("error") is None
    assert (tmp_path / ".pulsing" / "cluster.json").is_file()
    assert (tmp_path / ".pulsing" / "history" / "HEAD").is_file()
