# SPDX-License-Identifier: Apache-2.0
"""Workspace world view and puzzles."""

from __future__ import annotations

from pathlib import Path

from pulsing.agent.workspace.config import default_config, load_config, save_config
from pulsing.agent.workspace.quest import update_quest_status
from pulsing.agent.workspace.world_view import format_puzzles, puzzles_at, render_look


def test_default_has_puzzle() -> None:
    cfg = default_config(Path("/tmp/demo"))
    assert "unit-tests" in cfg.puzzles
    assert cfg.puzzles["unit-tests"].get("status") == "open"


def test_puzzles_at_tests_dir(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    cfg = default_config(tmp_path)
    here = puzzles_at(cfg, tmp_path / "tests")
    assert any(pid == "unit-tests" for pid, _ in here)


def test_render_look(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    cfg = default_config(tmp_path)
    text = render_look(cfg, cwd=tmp_path / "tests")
    assert "unit-tests" in text
    assert "player:" in text


def test_format_puzzles() -> None:
    cfg = default_config(Path("/tmp/demo"))
    assert "unit-tests" in format_puzzles(cfg, all_=True)


def test_quest_update_and_assign(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    save_config(cfg)
    updated = update_quest_status(
        tmp_path, "unit-tests", status="in_progress", reporter="guide"
    )
    assert updated["status"] == "in_progress"
    cfg2 = load_config(tmp_path)
    cfg2.puzzles["unit-tests"]["assign_to"] = "guide"
    save_config(cfg2)
    cfg3 = load_config(tmp_path)
    assert cfg3.puzzles["unit-tests"]["assign_to"] == "guide"
