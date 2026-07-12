# SPDX-License-Identifier: Apache-2.0
"""Dashboard layout generation."""

from __future__ import annotations

from pathlib import Path

from pulsing.cli.agent.commands.dashboard import (
    agent_cli_argv,
    find_backend,
    write_zellij_layout,
)
from pulsing.agent.workspace.config import default_config


def test_agent_cli_argv_includes_subcommand() -> None:
    argv = agent_cli_argv("watch", "-f")
    assert argv[-2:] == ["-f"] or "watch" in argv


def test_write_zellij_layout(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    cfg.default_agents = ["bard", "smith"]
    path = write_zellij_layout(cfg, interval=0.8)
    text = path.read_text(encoding="utf-8")
    assert "layout {" in text
    assert str(tmp_path.resolve()) in text
    assert 'name="bard"' in text
    assert 'name="smith"' in text
    assert "logs bard" in text


def test_write_zellij_demo_layout(tmp_path: Path) -> None:
    from pulsing.cli.agent.commands.dashboard import write_zellij_demo_layout

    cfg = default_config(tmp_path)
    path = write_zellij_demo_layout(
        cfg,
        demo_shell="export PCRaft=1; pulsing-agent demo --no-dashboard",
        agent_names=["bard", "smith", "sage"],
    )
    text = path.read_text(encoding="utf-8")
    assert 'name="demo"' in text
    assert 'name="bard"' in text
    assert 'name="smith"' in text
    assert 'name="sage"' in text
    assert "logs" in text


def test_zellij_session_line_parser() -> None:
    from pulsing.cli.agent.commands.dashboard import _parse_zellij_session_line

    line = "\x1b[32;1mpulsing-agent-demo\x1b[m [Created 1m ago] (\x1b[31;1mEXITED\x1b[m - attach to resurrect)"
    assert _parse_zellij_session_line(line, "pulsing-agent-demo") == "exited"
    assert _parse_zellij_session_line(line, "other") is None
    active = "pulsing-agent-demo [Created 1m ago] (ACTIVE)"
    assert _parse_zellij_session_line(active, "pulsing-agent-demo") == "active"


def test_zellij_session_constant() -> None:
    from pulsing.cli.agent.commands.dashboard import ZELLIJ_SESSION

    assert ZELLIJ_SESSION == "pulsing-agent-dashboard"


def test_find_backend_auto_or_skip() -> None:
    import shutil

    if shutil.which("zellij"):
        assert find_backend("auto") == "zellij"
    elif shutil.which("tmux"):
        assert find_backend("auto") == "tmux"


def test_demo_worker_shell() -> None:
    from argparse import Namespace

    from pulsing.cli.agent.commands.demo import demo_worker_shell
    from pulsing.cli.agent.commands.dashboard import DEMO_WORKER_ENV

    sh = demo_worker_shell(Namespace(interval=3.0, real_llm=False, addr="127.0.0.1:0"))
    assert DEMO_WORKER_ENV in sh
    assert "--no-dashboard" in sh
