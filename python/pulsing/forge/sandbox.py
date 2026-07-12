# SPDX-License-Identifier: Apache-2.0
"""Bash sandbox strategies (aligned with Codex/Craft MVP)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def normalize_policy(raw: str | None) -> str:
    p = (raw or "off").strip().lower()
    if p in ("0", "false", "none", ""):
        return "off"
    if p not in ("off", "restricted", "bwrap"):
        return "off"
    return p


def build_bash_exec(
    command: str,
    *,
    cwd: str | None,
    policy: str,
    dangerously_disable_sandbox: bool = False,
    timeout: int,
    login: bool = False,
) -> tuple[list[str], dict[str, str] | None, str]:
    """Build argv/env for `command` under `policy`.

    `login` requests login-shell semantics (`bash -l`, profile sourcing)
    instead of a plain `-c` invocation. It must never disable the sandbox
    wrapper itself; only `dangerously_disable_sandbox` may do that, and
    callers are expected to gate it behind approval.
    """
    _ = timeout
    wd = str(Path(cwd or ".").resolve())
    pol = "off" if dangerously_disable_sandbox else normalize_policy(policy)
    shell_flag = "-lc" if login else "-c"

    if pol == "off":
        argv = ["bash", "-lc", command] if login else ["/bin/sh", "-c", command]
        return (argv, None, "subprocess shell (sandbox=off)")

    if pol == "bwrap" and shutil.which("bwrap"):
        argv: list[str] = [
            "bwrap",
            "--die-with-parent",
            "--unshare-pid",
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
        ]
        if Path("/lib64").is_dir():
            argv.extend(["--ro-bind", "/lib64", "/lib64"])
        argv.extend(
            [
                "--bind",
                wd,
                "/work",
                "--chdir",
                "/work",
                "bash",
                shell_flag,
                command,
            ],
        )
        return (argv, None, "bubblewrap (minimal profile; Linux)")

    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "USER": os.environ.get("USER", "user"),
    }
    env_argv = ["env", "-i", *[f"{k}={v}" for k, v in env.items()]]
    if login:
        argv = [*env_argv, "bash", "-lc", command]
        label = "restricted env (env -i + bash -l)"
    else:
        argv = [*env_argv, "bash", "--norc", "--noprofile", "-c", command]
        label = "restricted env (env -i + bash --norc)"
    return (argv, env, label)
