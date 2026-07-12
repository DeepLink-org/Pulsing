# SPDX-License-Identifier: Apache-2.0
"""``pulsing agent`` CLI parser and world commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.cli.agent.commands.world import run_init, run_look
from pulsing.cli.agent.main import DEFAULT_PROG, _parse as parse


def test_parse_world_commands() -> None:
    assert parse(["init"]).cmd == "init"
    assert parse(["look"]).cmd == "look"
    assert parse(["demo"]).cmd == "demo"
    assert parse(["watch"]).cmd == "watch"
    assert parse(["dashboard"]).cmd == "dashboard"
    assert parse([]).cmd == "look"


def test_parse_agent_and_task() -> None:
    assert parse(["list"]).cmd == "list"
    assert parse(["say", "guide", "hi"]).cmd == "say"
    assert parse(["task", "list"]).task_cmd == "list"
    assert parse(["task", "show", "unit-tests"]).id == "unit-tests"


def test_parse_group_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        parse(["task"])


def test_parse_sleep_and_mark() -> None:
    assert parse(["sleep"]).cmd == "sleep"
    args = parse(["task", "mark", "unit-tests", "--status", "open"])
    assert args.task_cmd == "mark"
    assert args.assign_to is None


@pytest.mark.asyncio
async def test_init_and_look(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = parse(["init"], prog=DEFAULT_PROG)
    assert args.cmd == "init"
    await run_init(args)
    assert (tmp_path / ".pulsing" / "cluster.json").is_file()
    assert (tmp_path / ".pulsing" / "npcs" / "artisan.json").is_file()
    await run_look(parse(["look"]))
    await run_init(parse(["init"]))
