# SPDX-License-Identifier: Apache-2.0
"""Shell execpolicy + approval tests."""

from __future__ import annotations

import pytest

from pulsing.forge.permissions import PermissionChecker
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE


def test_permission_checker_auto_approves_exec() -> None:
    checker = PermissionChecker(auto_approve=True)
    assert (
        checker.prompt_exec_approval({"command": ["git", "reset", "--hard"]})
        == "approved"
    )


def test_permission_checker_denies_exec_without_callback() -> None:
    checker = PermissionChecker(auto_approve=False)
    assert checker.prompt_exec_approval({"command": ["echo", "x"]}) == "denied"


def test_permission_checker_exec_callback() -> None:
    seen: list[dict] = []

    def _cb(req: dict) -> str:
        seen.append(req)
        return "approved"

    checker = PermissionChecker(auto_approve=False, exec_approval_callback=_cb)
    assert (
        checker.prompt_exec_approval({"command": ["curl", "example.com"]}) == "approved"
    )
    assert seen[0]["command"] == ["curl", "example.com"]


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_rust_forbidden_git_reset_hard() -> None:
    from pulsing.forge.rust_runtime import RustForgeAdapter

    host = RustForgeAdapter.create(
        cwd=".",
        auto_approve=True,
    )
    out = host.call_tool("shell_command", {"command": "git reset --hard HEAD"})
    assert out.is_error
    assert "forbidden" in out.content.lower()


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_rust_exec_approval_blocks_without_callback() -> None:
    from pulsing.forge.rust_runtime import RustForgeAdapter

    host = RustForgeAdapter.create(cwd=".", auto_approve=False)
    out = host.call_tool("shell_command", {"command": "echo hello"})
    assert out.is_error
