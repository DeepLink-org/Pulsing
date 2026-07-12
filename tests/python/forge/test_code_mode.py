# SPDX-License-Identifier: Apache-2.0
"""``exec`` / code_mode behavior — sandbox boundary, cell state, error mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.testing.forge_harness import local_runtime

pytestmark = [pytest.mark.forge]


def _exec(rt, source: str, **extra):
    return rt.call_tool("exec", {"source": source, **extra})


def test_exec_runs_and_returns_text(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = _exec(rt, "text('hi')")
    assert not out.is_error
    assert "hi" in out.content
    assert out.structured["kind"] == "result"


def test_exec_empty_source_is_error(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("exec", {"source": "   "})
    assert out.is_error
    assert "empty" in out.content


def test_exec_error_message_includes_exception_type(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = _exec(rt, "1 / 0")
    assert out.structured["error_text"].startswith("ZeroDivisionError:")
    assert "Error:" in out.content


def test_exec_syntax_error_is_reported_not_raised(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = _exec(rt, "def broken(:\n    pass")
    assert out.structured["error_text"].startswith("SyntaxError:")


@pytest.mark.parametrize(
    "source",
    [
        "open('/etc/passwd')",
        "__import__('os')",
        "eval('1')",
        "exec('1')",
    ],
)
def test_exec_blocks_raw_io_and_import_builtins(
    forge_workspace: Path, source: str
) -> None:
    """Defense-in-depth: cells run in-process with no OS sandbox, so the most
    obvious escapes (file I/O, module import, eval/exec) must be unavailable
    via the restricted builtins — even though this is not a full security
    boundary (see code_mode/cell.py comment)."""
    rt = local_runtime(forge_workspace)
    out = _exec(rt, source)
    assert out.structured["error_text"].startswith("NameError:")


def test_exec_nested_tool_call_respects_allowlist(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = _exec(rt, "tools.call('exec', {'source': 'text(1)'})")
    assert out.structured["error_text"].startswith("PermissionError:")
    assert "not available in code mode" in out.structured["error_text"]


def test_exec_nested_tool_call_reaches_real_tool(forge_workspace: Path) -> None:
    (forge_workspace / "sample.txt").write_text("hello\n", encoding="utf-8")
    rt = local_runtime(forge_workspace)
    out = _exec(rt, "text(tools.call('shell_command', {'command': 'echo ok'}))")
    assert not out.is_error
    assert "ok" in out.content.lower()


def test_exec_rejects_sandbox_policy_when_isolation_requested(
    forge_workspace: Path,
) -> None:
    """``exec`` has no OS-level isolation to apply sandbox_policy to; it must
    fail closed instead of silently running unsandboxed code when isolation
    was explicitly requested (sandbox-boundary bypass otherwise)."""
    rt = local_runtime(forge_workspace, sandbox_policy="restricted")
    out = _exec(rt, "text('should not run')")
    assert out.is_error
    assert "sandbox_policy" in out.content


def test_exec_rejects_per_call_sandbox_off_when_ctx_restricted(
    forge_workspace: Path,
) -> None:
    """Per-call sandbox_policy='off' must not downgrade a restricted session."""
    rt = local_runtime(forge_workspace, sandbox_policy="restricted")
    out = rt.call_tool(
        "exec", {"source": "text('should not run')", "sandbox_policy": "off"}
    )
    assert out.is_error
    assert "sandbox_policy" in out.content


@pytest.mark.parametrize("policy", ["restricted", "bwrap"])
def test_exec_rejects_non_off_policies(forge_workspace: Path, policy: str) -> None:
    rt = local_runtime(forge_workspace, sandbox_policy=policy)
    out = _exec(rt, "text('should not run')")
    assert out.is_error
    assert policy in out.content


def test_exec_dangerously_disable_sandbox_overrides_rejection(
    forge_workspace: Path,
) -> None:
    rt = local_runtime(forge_workspace, sandbox_policy="restricted")
    out = _exec(rt, "text('ok')", dangerously_disable_sandbox=True)
    assert not out.is_error
    assert "ok" in out.content


def test_exec_off_policy_is_unaffected(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace, sandbox_policy="off")
    out = _exec(rt, "text('ok')")
    assert not out.is_error


def test_wait_unknown_cell_id_is_error(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("wait", {"cell_id": "does-not-exist"})
    assert out.is_error


def test_exec_then_wait_returns_same_cell(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    first = _exec(rt, "text('hi')")
    cell_id = first.structured["cell_id"]
    out = rt.call_tool("wait", {"cell_id": cell_id})
    assert not out.is_error
    assert out.structured["cell_id"] == cell_id


def test_exec_yield_then_wait_resumes(forge_workspace: Path) -> None:
    rt = local_runtime(forge_workspace)
    source = 'text("part1")\nyield_control()\ntext("part2")\n'
    first = _exec(rt, source)
    assert first.structured["kind"] == "yielded"
    assert "part1" in first.content
    assert "part2" not in first.content

    out = rt.call_tool("wait", {"cell_id": first.structured["cell_id"]})
    assert not out.is_error
    assert out.structured["kind"] == "result"
    assert "part2" in out.content
