# SPDX-License-Identifier: Apache-2.0
"""Tests for workflow scaffold and WorkflowContext."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.workspace.bootstrap import init_workspace
from pulsing.workflow import WorkflowContext, run


def test_init_scaffolds_workflows(tmp_path: Path) -> None:
    init_workspace(tmp_path, template="minimal", seed_npcs=False)
    workflows = tmp_path / ".pulsing" / "workflows"
    assert (workflows / "README.md").is_file()
    assert (workflows / "example.py").is_file()
    text = (workflows / "example.py").read_text(encoding="utf-8")
    assert "WorkflowContext" in text
    assert "pulsing.workflow" in text


def test_workflow_context_checkpoint(tmp_path: Path) -> None:
    init_workspace(tmp_path, template="minimal", seed_npcs=False)
    (tmp_path / "note.txt").write_text("v1", encoding="utf-8")

    ctx = WorkflowContext(root=tmp_path)
    rev = ctx.checkpoint("workflow step")
    assert rev == "0002"  # 0001 is workspace init

    (tmp_path / "note.txt").write_text("v2", encoding="utf-8")
    rolled = ctx.rollback()
    assert rolled
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "v1"


def test_workflow_run_sync(tmp_path: Path) -> None:
    init_workspace(tmp_path, template="minimal", seed_npcs=False)
    seen: list[str] = []

    def main(ctx: WorkflowContext) -> None:
        seen.append(str(ctx.root))

    run(main, root=tmp_path)
    assert seen == [str(tmp_path.resolve())]


def test_workflow_run_async(tmp_path: Path) -> None:
    init_workspace(tmp_path, template="minimal", seed_npcs=False)
    seen: list[str] = []

    async def main(ctx: WorkflowContext) -> None:
        seen.append("async")

    run(main, root=tmp_path)
    assert seen == ["async"]
