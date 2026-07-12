# SPDX-License-Identifier: Apache-2.0
"""Shared Forge test harness — minimal args, smoke runners, wire assertions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from pulsing.forge.integrated import (
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
    FORGE_TOOL_NAMES,
)
from pulsing.forge.result import ToolResult
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.session import LocalToolSession

# Tools implemented only on the Rust Hybrid path today (no Python handler).
RUST_ONLY_HOST_TOOLS: frozenset[str] = frozenset(
    {
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "request_permissions",
    }
)

LOCAL_PYTHON_TOOLS: frozenset[str] = FORGE_TOOL_NAMES - RUST_ONLY_HOST_TOOLS


def assert_forge_manifest() -> None:
    assert len(FORGE_TOOL_NAMES) == 32
    assert FORGE_TOOL_NAMES == FORGE_ISOLATED_TOOL_NAMES | FORGE_HOST_TOOL_NAMES
    assert not (FORGE_ISOLATED_TOOL_NAMES & FORGE_HOST_TOOL_NAMES)


# Minimal 1x1 PNG (same bytes used across Forge tests).
_MIN_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c630001000005000108d4a7e00000000049454e44ae426082"
)


class ForgeRuntime(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


def seed_workspace(tmp_path: Path) -> Path:
    """Create common files used by minimal tool args."""
    sample = tmp_path / "sample.txt"
    sample.write_text("hello\n", encoding="utf-8")
    png = tmp_path / "x.png"
    png.write_bytes(_MIN_PNG)
    return sample


def minimal_tool_args(name: str, tmp_path: Path) -> dict[str, Any]:
    """Arguments that exercise each registered tool without Unknown-tool failures."""
    seed_workspace(tmp_path)
    p = tmp_path / "sample.txt"
    patch = "*** Begin Patch\n*** Add File: added.txt\n+added\n*** End Patch\n"
    common: dict[str, dict[str, Any]] = {
        "Read": {"file_path": str(p)},
        "Glob": {"pattern": "*.txt", "path": str(tmp_path)},
        "Grep": {"pattern": "hello", "path": str(tmp_path)},
        "Edit": {"file_path": str(p), "old_string": "hello", "new_string": "hi"},
        "Write": {"file_path": str(tmp_path / "out.txt"), "content": "x"},
        "Bash": {"command": "echo ok"},
        "shell_command": {
            "command": "echo ok",
            "workdir": str(tmp_path),
            "timeout_ms": 5000,
        },
        "exec_command": {"cmd": "echo ok", "yield_time_ms": 300, "tty": False},
        "write_stdin": {"session_id": 0, "chars": ""},
        "apply_patch": {"patch": patch},
        "view_image": {"path": str(tmp_path / "x.png"), "detail": "low"},
        "update_plan": {"plan": [{"step": "s", "status": "pending"}]},
        "new_context": {},
        "get_context_remaining": {},
        "request_user_input": {
            "questions": [{"id": "q", "prompt": "p?", "type": "text"}]
        },
        "request_permissions": {"permissions": ["network"], "reason": "test"},
        "tool_search": {"query": "read"},
        "list_available_plugins_to_install": {},
        "request_plugin_install": {
            "tool_id": "test",
            "tool_name": "test",
            "suggest_reason": "test",
        },
        "list_mcp_resources": {},
        "list_mcp_resource_templates": {"server": "demo"},
        "read_mcp_resource": {"server": "demo", "uri": "file:///dev/null"},
        "exec": {"source": "print('hi')"},
        "wait": {"cell_id": "missing", "timeout_ms": 1},
        "web.run": {"url": "https://example.com"},
        "skills.list": {},
        "skills.read": {"name": "missing"},
        "memories.list": {},
        "memories.read": {"id": "missing"},
        "memories.search": {"query": "test"},
        "memories.add_ad_hoc_note": {"content": "note"},
        "web_search": {"query": "test"},
    }
    return common.get(name, {})


def assert_callable_result(name: str, out: ToolResult) -> None:
    if out.content.startswith("Unknown tool:"):
        raise AssertionError(f"{name}: {out.content}")


@dataclass(frozen=True)
class SmokeResult:
    tool: str
    ok: bool
    detail: str = ""


def run_tool_smoke(
    rt: ForgeRuntime,
    tmp_path: Path,
    *,
    tools: frozenset[str] | None = None,
) -> list[SmokeResult]:
    names = sorted(tools or FORGE_TOOL_NAMES)
    results: list[SmokeResult] = []
    for name in names:
        try:
            out = rt.call_tool(name, minimal_tool_args(name, tmp_path))
            if out.content.startswith("Unknown tool:"):
                results.append(SmokeResult(name, False, out.content))
            else:
                results.append(SmokeResult(name, True))
        except Exception as exc:  # noqa: BLE001 — aggregate smoke failures
            results.append(SmokeResult(name, False, str(exc)))
    return results


def smoke_failures(results: list[SmokeResult]) -> list[str]:
    return [f"{r.tool}: {r.detail or 'failed'}" for r in results if not r.ok]


def local_runtime(
    tmp_path: Path,
    *,
    session: LocalToolSession | None = None,
    sandbox_policy: str = "off",
    dangerously_disable_sandbox: bool = False,
) -> LocalToolRuntime:
    sess = session or LocalToolSession(token_budget=128_000)
    return LocalToolRuntime(
        cwd=str(tmp_path),
        session=sess,
        sandbox_policy=sandbox_policy,
        dangerously_disable_sandbox=dangerously_disable_sandbox,
    )


def session_from_runtime(rt: ForgeRuntime) -> LocalToolSession | None:
    if hasattr(rt, "session"):
        sess = rt.session  # type: ignore[attr-defined]
        return sess if isinstance(sess, LocalToolSession) else None
    if hasattr(rt, "python_runtime"):
        sess = rt.python_runtime.session  # type: ignore[attr-defined]
        return sess if isinstance(sess, LocalToolSession) else None
    return None


# L2 wire checks — return None when skipped, raise AssertionError on mismatch.
WireCheck = Callable[[ForgeRuntime, Path, ToolResult], None]

_WIRE_CHECKS: dict[str, WireCheck] = {}


def register_wire_check(tool: str, fn: WireCheck) -> None:
    _WIRE_CHECKS[tool] = fn


def wire_check_tools() -> frozenset[str]:
    return frozenset(_WIRE_CHECKS)


def run_wire_check(rt: ForgeRuntime, tmp_path: Path, tool: str) -> None:
    fn = _WIRE_CHECKS.get(tool)
    if fn is None:
        return
    out = rt.call_tool(tool, minimal_tool_args(tool, tmp_path))
    assert_callable_result(tool, out)
    fn(rt, tmp_path, out)


def _check_read(_rt: ForgeRuntime, tmp_path: Path, out: ToolResult) -> None:
    assert not out.is_error
    assert out.content.strip() == "hello"


def _check_apply_patch(_rt: ForgeRuntime, tmp_path: Path, out: ToolResult) -> None:
    assert not out.is_error
    assert (tmp_path / "added.txt").read_text(encoding="utf-8") == "added\n"


def _check_shell_command(_rt: ForgeRuntime, _tmp_path: Path, out: ToolResult) -> None:
    assert not out.is_error
    assert "ok" in out.content.lower()


def _check_update_plan(rt: ForgeRuntime, _tmp_path: Path, out: ToolResult) -> None:
    assert not out.is_error
    sess = session_from_runtime(rt)
    if sess is not None and len(sess.plan) >= 1:
        return
    assert "plan" in out.content.lower()


def _check_get_context_remaining(
    _rt: ForgeRuntime, _tmp_path: Path, out: ToolResult
) -> None:
    assert not out.is_error
    payload = out.structured if out.structured is not None else json.loads(out.content)
    assert payload["status"] in ("ok", "unknown")
    if payload["status"] == "ok":
        assert isinstance(payload["tokens_remaining"], int)
        assert payload["tokens_remaining"] >= 0
    else:
        assert payload["tokens_remaining"] is None


def _check_new_context(rt: ForgeRuntime, _tmp_path: Path, out: ToolResult) -> None:
    assert not out.is_error
    sess = session_from_runtime(rt)
    if sess is not None and sess.new_context_requested:
        return
    assert "context" in out.content.lower()


def _check_list_mcp_resources(
    _rt: ForgeRuntime, _tmp_path: Path, out: ToolResult
) -> None:
    assert not out.is_error
    import json

    parsed = json.loads(out.content)
    assert isinstance(parsed, dict)


register_wire_check("list_mcp_resources", _check_list_mcp_resources)
register_wire_check("Read", _check_read)
register_wire_check("apply_patch", _check_apply_patch)
register_wire_check("shell_command", _check_shell_command)
register_wire_check("update_plan", _check_update_plan)
register_wire_check("get_context_remaining", _check_get_context_remaining)
register_wire_check("new_context", _check_new_context)
