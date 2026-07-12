# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge session REPL."""

from __future__ import annotations

import json
from pathlib import Path

from pulsing.forge.repl.parse import parse_line
from pulsing.forge.repl.session import ForgeReplSession
from pulsing.forge.repl.trace import TraceLog, TraceRecord, load_trace, save_trace


def test_parse_bare_tool_with_json() -> None:
    cmd, args = parse_line('Read {"file_path": "README.md"}')
    assert cmd == "call"
    assert args["tool"] == "Read"
    assert args["arguments"]["file_path"] == "README.md"


def test_parse_nu_flags() -> None:
    cmd, args = parse_line("Glob --pattern *.py")
    assert cmd == "call"
    assert args["tool"] == "Glob"
    assert args["arguments"]["pattern"] == "*.py"


def test_trace_roundtrip(tmp_path: Path) -> None:
    log = TraceLog()
    log.append(
        TraceRecord(
            seq=1,
            kind="tool_call",
            tool="Read",
            arguments={"file_path": "a.txt"},
            result={"content": "hi", "is_error": False},
        )
    )
    path = tmp_path / "t.jsonl"
    save_trace(path, log)
    loaded = load_trace(path)
    assert len(loaded.tool_calls()) == 1
    assert loaded.tool_calls()[0].tool == "Read"


def test_repl_call_read(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("world", encoding="utf-8")
    session = ForgeReplSession(cwd=tmp_path)
    out = session.call_tool("Read", {"file_path": "hello.txt"})
    assert not out.is_error
    assert out.content == "world"


def test_replay_dry_run(tmp_path: Path) -> None:
    log = TraceLog()
    log.append(
        TraceRecord(
            seq=1,
            kind="tool_call",
            tool="Read",
            arguments={"file_path": "x"},
            result={"content": "", "is_error": True},
        )
    )
    path = tmp_path / "replay.jsonl"
    save_trace(path, log)
    session = ForgeReplSession(cwd=tmp_path)
    session.load_replay_trace(path)
    msg = session.replay_step(dry_run=True)
    assert "dry-run" in msg
    assert "Read" in msg


def test_replay_verify_read(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("same", encoding="utf-8")
    log = TraceLog()
    log.append(
        TraceRecord(
            seq=1,
            kind="tool_call",
            tool="Read",
            arguments={"file_path": "a.txt"},
            result={"content": "same", "is_error": False},
        )
    )
    path = tmp_path / "v.jsonl"
    save_trace(path, log)
    session = ForgeReplSession(cwd=tmp_path)
    session.load_replay_trace(path)
    msg = session.replay_step(verify=True)
    assert "[ok]" in msg


def test_session_table(tmp_path: Path) -> None:
    session = ForgeReplSession(cwd=tmp_path, approval_mode="ask")
    text = session.format_session_table()
    assert "approval" in text
    assert "ask" in text
