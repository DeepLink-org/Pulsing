from __future__ import annotations

from pathlib import Path

from pulsing.forge.sandbox import build_bash_exec, normalize_policy


def test_normalize_policy() -> None:
    assert normalize_policy(None) == "off"
    assert normalize_policy("restricted") == "restricted"
    assert normalize_policy("bwrap") == "bwrap"
    assert normalize_policy("bogus") == "off"


def test_build_bash_exec_off_uses_sh_c() -> None:
    argv, env, label = build_bash_exec(
        "echo hi",
        cwd="/tmp",
        policy="off",
        dangerously_disable_sandbox=False,
        timeout=5,
    )
    assert argv[:3] == ["/bin/sh", "-c", "echo hi"]
    assert env is None
    assert "sandbox=off" in label or "shell" in label


def test_effective_shell_policy_require_escalated() -> None:
    from pulsing.forge.context import ToolCallContext
    from pulsing.forge.handlers import _effective_shell_policy

    ctx = ToolCallContext(cwd=Path("."), sandbox_policy="restricted")
    args = {"sandbox_permissions": "require_escalated"}
    assert _effective_shell_policy(ctx, args, "restricted") == "off"


def test_login_restricted_policy_still_clears_env() -> None:
    """Regression: login=true must not bypass the restricted-env wrapper."""
    argv, env, label = build_bash_exec(
        "echo hi",
        cwd="/tmp",
        policy="restricted",
        dangerously_disable_sandbox=False,
        timeout=5,
        login=True,
    )
    assert argv[0] == "env"
    assert argv[1] == "-i"
    assert env is not None
    assert "-lc" in argv
    assert "restricted env" in label


def test_login_off_policy_uses_bash_lc() -> None:
    argv, env, label = build_bash_exec(
        "echo hi",
        cwd="/tmp",
        policy="off",
        dangerously_disable_sandbox=False,
        timeout=5,
        login=True,
    )
    assert argv == ["bash", "-lc", "echo hi"]
    assert env is None
    assert "sandbox=off" in label
