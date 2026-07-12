# SPDX-License-Identifier: Apache-2.0
"""Demo LLM + demo command."""

from __future__ import annotations

from pathlib import Path

from pulsing.cli.agent.commands.demo import (
    CHATTER_SCRIPT,
    DEMO_AGENTS,
    prepare_demo_workspace,
)
from pulsing.agent.loop.demo_llm import plan_demo_turn


def test_plan_demo_glob_tool() -> None:
    plan = plan_demo_turn(
        [{"role": "user", "content": "List project files with Glob under ."}],
        [{"name": "Glob", "input_schema": {}}],
    )
    assert plan["kind"] == "tool"
    assert plan["name"] == "Glob"


def test_plan_demo_text_fallback() -> None:
    plan = plan_demo_turn(
        [{"role": "user", "content": "hello world"}],
        [],
    )
    assert plan["kind"] == "text"
    assert "hello" in plan["text"]


def test_prepare_demo_workspace(tmp_path: Path) -> None:
    cfg = prepare_demo_workspace(tmp_path)
    assert cfg.shared_tool_worker is True
    assert cfg.default_agents == ["bard", "smith", "sage"]
    assert cfg.puzzles["unit-tests"]["assign_to"] == "sage"
    assert (tmp_path / ".pulsing" / "cluster.json").is_file()


def test_chatter_script_has_three_agents() -> None:
    names = {a[0] for a in DEMO_AGENTS}
    for a, b, _ in CHATTER_SCRIPT:
        assert a in names
        assert b in names
