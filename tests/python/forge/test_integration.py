# SPDX-License-Identifier: Apache-2.0
"""L3 Forge integration — trace replay and end-to-end tool flows."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.repl.session import ForgeReplSession
from pulsing.forge.repl.trace import TraceLog, TraceRecord, load_trace, save_trace

pytestmark = [pytest.mark.forge, pytest.mark.forge_l3]

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "forge_traces"


@pytest.mark.forge_l4
def test_l4_trace_fixture_replay_verify(tmp_path: Path) -> None:
    src = tmp_path / "sample.txt"
    src.write_text("forge-repl-fixture", encoding="utf-8")
    log = TraceLog()
    log.append(
        TraceRecord(
            seq=1,
            kind="tool_call",
            tool="Read",
            arguments={"file_path": "sample.txt"},
            result={"content": "forge-repl-fixture", "is_error": False},
        )
    )
    trace_path = tmp_path / "t.jsonl"
    save_trace(trace_path, log)
    session = ForgeReplSession(cwd=tmp_path)
    session.load_replay_trace(trace_path)
    msg = session.replay_step(verify=True)
    assert "[ok]" in msg


def test_l3_shipped_trace_fixture() -> None:
    path = _FIXTURES / "read_sample.jsonl"
    assert path.is_file()
    loaded = load_trace(path)
    assert len(loaded.tool_calls()) == 1
    assert loaded.tool_calls()[0].tool == "Read"
