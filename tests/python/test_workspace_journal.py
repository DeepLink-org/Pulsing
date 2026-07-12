# SPDX-License-Identifier: Apache-2.0
"""Tests for workspace journal (Python Path A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.workspace.bootstrap import init_workspace
from pulsing.workspace.journal import checkpoint, current_head, list_revisions, rollback
from pulsing.workspace.layout import WorkspaceLayout


def test_init_checkpoint_rollback_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.txt").write_text("v1", encoding="utf-8")

    result = init_workspace(template="minimal", seed_npcs=False)
    assert result.created
    assert (tmp_path / ".pulsing" / "cluster.json").is_file()
    assert (tmp_path / ".pulsing" / "workspace.json").is_file()
    assert (tmp_path / ".pulsing" / "history" / "HEAD").is_file()

    layout = WorkspaceLayout(tmp_path)
    (tmp_path / "hello.txt").write_text("v2", encoding="utf-8")
    checkpoint(layout, message="v2")

    (tmp_path / "hello.txt").write_text("v3", encoding="utf-8")
    rollback(layout)
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "v2"

    revs = list_revisions(layout)
    assert len(revs) >= 2
    assert current_head(layout) == revs[-1].id


def test_init_idempotent(tmp_path: Path) -> None:
    first = init_workspace(tmp_path, template="minimal", seed_npcs=False)
    second = init_workspace(tmp_path, template="minimal", seed_npcs=False)
    assert first.created
    assert not second.created
