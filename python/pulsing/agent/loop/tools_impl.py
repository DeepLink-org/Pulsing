# SPDX-License-Identifier: Apache-2.0
"""Pure tool implementations returning :class:`ToolResult` (shared by agent + worker)."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pulsing.agent.paths import agent_env
from pulsing.agent.loop.sandbox import build_bash_exec, normalize_policy
from pulsing.agent.loop.tool_base import ToolResult

_READ_CAP = 2 * 1024 * 1024
_BASH_MAX_OUT = 256 * 1024
_GREP_MAX = 200


def impl_read(**kwargs: Any) -> ToolResult:
    path = Path(str(kwargs.get("file_path", "")))
    try:
        data = path.read_bytes()
    except OSError as e:
        return ToolResult(content=str(e), is_error=True)
    if len(data) > _READ_CAP:
        return ToolResult(content="File too large for Read tool.", is_error=True)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return ToolResult(content="Not valid UTF-8.", is_error=True)
    return ToolResult(content=text)


def impl_glob(**kwargs: Any) -> ToolResult:
    pattern = str(kwargs.get("pattern", ""))
    base = Path(str(kwargs.get("path", "."))).resolve()
    if not base.exists():
        return ToolResult(content="path does not exist", is_error=True)
    try:
        matches = sorted(str(p) for p in base.glob(pattern))[:500]
    except OSError as e:
        return ToolResult(content=str(e), is_error=True)
    return ToolResult(content="\n".join(matches) if matches else "(no matches)")


def impl_grep(**kwargs: Any) -> ToolResult:
    raw_pat = str(kwargs.get("pattern", ""))
    root = Path(str(kwargs.get("path", "."))).resolve()
    glob_pat = kwargs.get("glob")
    try:
        cre = re.compile(raw_pat)
    except re.error as e:
        return ToolResult(content=f"Invalid regex: {e}", is_error=True)
    hits: list[str] = []

    def consider_file(fp: Path) -> None:
        if glob_pat and not fnmatch.fnmatch(fp.name, str(glob_pat)):
            return
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for i, line in enumerate(text.splitlines(), 1):
            if cre.search(line):
                hits.append(f"{fp}:{i}:{line[:500]}")
                if len(hits) >= _GREP_MAX:
                    return

    if not root.exists():
        return ToolResult(content="path not found", is_error=True)
    if root.is_file():
        consider_file(root)
    else:
        for fp in root.rglob("*"):
            if fp.is_file() and len(hits) < _GREP_MAX:
                consider_file(fp)
            if len(hits) >= _GREP_MAX:
                break
    if not hits:
        return ToolResult(content="(no matches)")
    extra = "\n… truncated …" if len(hits) >= _GREP_MAX else ""
    return ToolResult(content="\n".join(hits) + extra)


def impl_edit(**kwargs: Any) -> ToolResult:
    fp = Path(str(kwargs.get("file_path", "")))
    old = str(kwargs.get("old_string", ""))
    new = str(kwargs.get("new_string", ""))
    try:
        text = fp.read_text(encoding="utf-8")
    except OSError as e:
        return ToolResult(content=str(e), is_error=True)
    count = text.count(old)
    if count == 0:
        return ToolResult(content="old_string not found", is_error=True)
    if count > 1:
        return ToolResult(
            content="old_string is not unique; refusing ambiguous edit",
            is_error=True,
        )
    updated = text.replace(old, new, 1)
    try:
        fp.write_text(updated, encoding="utf-8")
    except OSError as e:
        return ToolResult(content=str(e), is_error=True)
    return ToolResult(content="ok")


def impl_write(**kwargs: Any) -> ToolResult:
    fp = Path(str(kwargs.get("file_path", "")))
    content = str(kwargs.get("content", ""))
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
    except OSError as e:
        return ToolResult(content=str(e), is_error=True)
    return ToolResult(content="ok")


def impl_bash(**kwargs: Any) -> ToolResult:
    cmd = str(kwargs.get("command", ""))
    timeout = int(kwargs.get("timeout_sec", 120))
    cwd = kwargs.get("cwd")
    if cwd is not None:
        cwd = str(cwd)
    policy = normalize_policy(str(kwargs.get("sandbox_policy", "off")))
    dangerous = bool(kwargs.get("dangerously_disable_sandbox", False))
    argv, extra_env, label = build_bash_exec(
        cmd,
        cwd=cwd,
        policy=policy,
        dangerously_disable_sandbox=dangerous,
        timeout=timeout,
    )
    run_kw: dict[str, Any] = {
        "args": argv,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }
    if cwd:
        run_kw["cwd"] = cwd
    if extra_env is not None:
        run_kw["env"] = extra_env
    try:
        proc = subprocess.run(**run_kw)
    except subprocess.TimeoutExpired:
        return ToolResult(content="timed out", is_error=True)
    except Exception as e:
        return ToolResult(content=str(e), is_error=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > _BASH_MAX_OUT:
        out = out[:_BASH_MAX_OUT] + "\n… truncated …"
    tail = f"\nexit={proc.returncode}\n[{label}]"
    return ToolResult(content=out + tail, is_error=proc.returncode != 0)


_FETCH_CAP = 256 * 1024


def impl_fetch_url(**kwargs: Any) -> ToolResult:
    """HTTP(S) GET with host allowlist (``PULSING_CRAFT_FETCH_ALLOW`` comma-separated hostnames)."""

    url = str(kwargs.get("url", "")).strip()
    max_bytes = int(kwargs.get("max_bytes", _FETCH_CAP))
    max_bytes = max(1024, min(max_bytes, _FETCH_CAP))
    if not url:
        return ToolResult(content="url required", is_error=True)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ToolResult(content="only http/https URLs", is_error=True)
    host = (parsed.hostname or "").lower()
    if not host:
        return ToolResult(content="missing host", is_error=True)
    allow = agent_env("FETCH_ALLOW").strip()
    if not allow:
        return ToolResult(
            content=(
                "FetchUrl disabled: set env PULSING_CRAFT_FETCH_ALLOW to a comma-separated "
                "hostname allowlist (e.g. api.github.com,example.com)."
            ),
            is_error=True,
        )
    allowed = {h.strip().lower() for h in allow.split(",") if h.strip()}
    if host not in allowed:
        return ToolResult(
            content=f"host {host!r} not in PULSING_CRAFT_FETCH_ALLOW",
            is_error=True,
        )
    req = Request(url, headers={"User-Agent": "pulsing-craft-fetch/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310 — host allowlisted
            data = resp.read(max_bytes + 1)
    except HTTPError as e:
        return ToolResult(content=f"HTTP {e.code}: {e.reason}", is_error=True)
    except URLError as e:
        return ToolResult(content=str(e.reason), is_error=True)
    except Exception as e:
        return ToolResult(content=str(e), is_error=True)
    if len(data) > max_bytes:
        return ToolResult(
            content="response larger than max_bytes",
            is_error=True,
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return ToolResult(content=text)
