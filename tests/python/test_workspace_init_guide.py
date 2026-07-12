# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM-guided ``pulsing init``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pulsing.workspace.bootstrap import init_workspace
from pulsing.workspace.init_guide import run_init_guide


@pytest.mark.asyncio
async def test_init_stores_guide_in_manifest(tmp_path: Path) -> None:
    init_workspace(
        tmp_path,
        template="minimal",
        seed_npcs=False,
        guide="Python ML project with pytest",
        provider="demo",
        model="demo",
    )
    data = json.loads((tmp_path / ".pulsing" / "workspace.json").read_text())
    assert data.get("init_guide") == "Python ML project with pytest"


@pytest.mark.asyncio
async def test_init_guide_demo_writes_readme(tmp_path: Path) -> None:
    init_workspace(tmp_path, template="minimal", seed_npcs=False, guide=None)
    await run_init_guide(
        tmp_path,
        "Create README.md describing a minimal Python CLI tool with pytest",
        provider="demo",
        model="demo",
    )
    readme = tmp_path / "README.md"
    assert readme.is_file()
    assert readme.read_text(encoding="utf-8")
