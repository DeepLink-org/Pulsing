# SPDX-License-Identifier: Apache-2.0
"""Built-in tool implementations for pulsing.forge."""

from __future__ import annotations

import base64
import errno
import fnmatch
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pulsing.forge.context import ToolCallContext, resolve_within_cwd
from pulsing.forge.exec_output import SHELL_MAX_BYTES, shell_timeout_ms
from pulsing.forge.patch_invocation import (
    MaybeApplyPatch,
    ParsedPatch,
    apply_parsed_patch,
    maybe_parse_apply_patch,
)
from pulsing.forge.result import ToolResult
from pulsing.forge.sandbox import build_bash_exec, normalize_policy
from pulsing.forge.discovery.catalog import ToolCatalogRefreshError
from pulsing.forge.discovery.entries import (
    TOOL_SEARCH_MAX_QUERY_CHARS,
    normalize_tool_search_limit,
)
from pulsing.forge.discovery.discoverable import (
    DiscoverablePlugin,
    DiscoverableToolAction,
    DiscoverableToolType,
    build_plugin_install_elicitation_meta,
)
from pulsing.forge.discovery.install import execute_plugin_install
from pulsing.forge.grep_worker import GREP_MAX as _GREP_MAX
from pulsing.forge.grep_worker import worker as _grep_scan_worker
from pulsing.forge.session import UpdatePlanArgs
from pulsing.forge.session_input import args_to_payload, validate_request_user_input

_READ_CAP = 2 * 1024 * 1024
_GREP_PATTERN_MAX = 1000
_GREP_TIMEOUT_SEC = 5.0
_GLOB_MAX = 500
_VIEW_IMAGE_CAP = 8 * 1024 * 1024
_HIGH_DETAIL_MAX_PX = 2048

# Must stay byte-for-byte identical to NEW_CONTEXT_MESSAGE in
# crates/pulsing-forge/src/handlers/session.rs — this pure-Python fallback path
# does not link against that crate, so the string is duplicated rather than shared.
NEW_CONTEXT_MESSAGE = (
    "A new context window will start without summarizing conversation history."
)
# Must stay byte-for-byte identical to PLAN_UPDATED in
# crates/pulsing-forge/src/handlers/plan.rs — this pure-Python fallback path
# does not link against that crate, so the string is duplicated rather than shared.
PLAN_UPDATED = "Plan updated"

_CODEX_SHELL = frozenset({"shell_command", "exec_command", "write_stdin"})
_CODEX_FILE = frozenset({"apply_patch", "view_image"})
_CODEX_SESSION = frozenset(
    {"update_plan", "new_context", "get_context_remaining", "request_user_input"}
)
_CODEX_DISCOVERY = frozenset(
    {"tool_search", "list_available_plugins_to_install", "request_plugin_install"}
)
_CODEX_CODE_MODE = frozenset({"exec", "wait"})
from pulsing.forge.extension.protocol import EXTENSION_TOOL_NAMES as _CODEX_EXTENSION

_CLAUDE_STYLE = frozenset({"Read", "Glob", "Grep", "Edit", "Write", "Bash"})

_ALL = (
    _CODEX_SHELL
    | _CODEX_FILE
    | _CODEX_SESSION
    | _CODEX_DISCOVERY
    | _CODEX_CODE_MODE
    | _CODEX_EXTENSION
    | _CLAUDE_STYLE
)


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    ctx: ToolCallContext,
) -> ToolResult:
    if name in _CODEX_EXTENSION:
        from pulsing.forge.extension.handlers import dispatch_extension_tool

        return dispatch_extension_tool(name, arguments, ctx=ctx)
    if name not in _ALL:
        return ToolResult(content=f"Unknown tool: {name}", is_error=True)
    impl = {
        "shell_command": _shell_command,
        "exec_command": _exec_command,
        "write_stdin": _write_stdin,
        "apply_patch": _apply_patch,
        "view_image": _view_image,
        "update_plan": _update_plan,
        "new_context": _new_context,
        "get_context_remaining": _get_context_remaining,
        "request_user_input": _request_user_input,
        "tool_search": _tool_search,
        "list_available_plugins_to_install": _list_available_plugins,
        "request_plugin_install": _request_plugin_install,
        "exec": _exec,
        "wait": _wait,
        "Read": _read,
        "Glob": _glob,
        "Grep": _grep,
        "Edit": _edit,
        "Write": _write,
        "Bash": _shell_command,
    }[name]
    kwargs = dict(arguments)
    kwargs.setdefault("cwd", str(ctx.cwd))
    kwargs.setdefault("sandbox_policy", ctx.sandbox_policy)
    kwargs.setdefault("dangerously_disable_sandbox", ctx.dangerously_disable_sandbox)
    return impl(ctx=ctx, **kwargs)


def _resolve_path(ctx: ToolCallContext, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ctx.cwd / p


def _read_os_error(path: Path, exc: OSError) -> str:
    """Mirror `read_error` in crates/pulsing-forge/src/handlers/read.rs."""
    if exc.errno == errno.ENOENT:
        reason = "No such file"
    elif exc.errno in (errno.EACCES, errno.EPERM):
        reason = "Permission denied"
    else:
        reason = str(exc)
    return f"{reason}: {path}"


def _read(
    *,
    ctx: ToolCallContext,
    file_path: str,
    offset: int | None = None,
    limit: int | None = None,
    **_: Any,
) -> ToolResult:
    path = _resolve_path(ctx, file_path)
    if path.is_dir():
        return ToolResult(
            content=f"Path is a directory, not a file: {path}", is_error=True
        )
    if offset is not None or limit is not None:
        return _read_range(path, max(offset or 1, 1), limit)
    try:
        data = path.read_bytes()
    except OSError as e:
        return ToolResult(content=_read_os_error(path, e), is_error=True)
    if len(data) > _READ_CAP:
        return ToolResult(
            content=(
                f"File too large for Read tool ({len(data)} bytes > {_READ_CAP} cap); "
                "retry with offset/limit to page through it."
            ),
            is_error=True,
        )
    try:
        return ToolResult(content=data.decode("utf-8"))
    except UnicodeDecodeError:
        return ToolResult(content=f"Not valid UTF-8: {path}", is_error=True)


def _read_range(path: Path, start_line: int, limit: int | None) -> ToolResult:
    """Streams `path` line by line so a paginated read never needs the whole file in memory."""
    end_line = start_line + limit if limit is not None else None
    out: list[str] = []
    size = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if lineno < start_line:
                    continue
                if end_line is not None and lineno >= end_line:
                    break
                size += len(line)
                if size > _READ_CAP:
                    return ToolResult(
                        content="Requested range too large for Read tool; use a smaller limit.",
                        is_error=True,
                    )
                out.append(line)
    except UnicodeDecodeError:
        return ToolResult(content=f"Not valid UTF-8: {path}", is_error=True)
    except OSError as e:
        return ToolResult(content=str(e), is_error=True)
    return ToolResult(content="".join(out))


def _glob(
    *, ctx: ToolCallContext, pattern: str, path: str | None = None, **_: Any
) -> ToolResult:
    if Path(pattern).is_absolute():
        return ToolResult(
            content="pattern must be relative to path/cwd; absolute glob patterns are not supported",
            is_error=True,
        )
    base = _resolve_path(ctx, path) if path else ctx.cwd
    if not base.exists():
        return ToolResult(content=f"path does not exist: {base}", is_error=True)
    try:
        matches = sorted(str(p) for p in base.glob(pattern))
    except (OSError, NotImplementedError, ValueError) as e:
        return ToolResult(
            content=f"invalid glob pattern {pattern!r}: {e}", is_error=True
        )
    if not matches:
        return ToolResult(content="(no matches)")
    total = len(matches)
    truncated = matches[:_GLOB_MAX]
    if total > _GLOB_MAX:
        truncated.append(f"… truncated: showing {_GLOB_MAX} of {total} matches …")
    return ToolResult(content="\n".join(truncated))


def _grep_boundary(ctx: ToolCallContext, root: Path) -> Path | None:
    """When search root is under cwd, skip files that resolve outside cwd."""
    try:
        root.resolve().relative_to(ctx.cwd.resolve())
    except ValueError:
        return None
    return ctx.cwd


def _grep(
    *,
    ctx: ToolCallContext,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    **_: Any,
) -> ToolResult:
    if len(pattern) > _GREP_PATTERN_MAX:
        return ToolResult(
            content=f"Pattern too long ({len(pattern)} > {_GREP_PATTERN_MAX} chars); simplify the regex",
            is_error=True,
        )
    root = _resolve_path(ctx, path) if path else ctx.cwd
    if not root.exists():
        return ToolResult(content="path not found", is_error=True)
    try:
        re.compile(pattern)
    except re.error as e:
        return ToolResult(content=f"Invalid regex: {e}", is_error=True)

    # `re` has no backtracking budget; run the scan in a child process so a
    # pathological pattern cannot block the tool runtime indefinitely.
    boundary = _grep_boundary(ctx, root)
    ctx_mp = multiprocessing.get_context("spawn")
    q: Any = ctx_mp.Queue()
    proc = ctx_mp.Process(
        target=_grep_scan_worker,
        args=(str(root), pattern, glob, str(boundary) if boundary else None, q),
    )
    proc.start()
    proc.join(timeout=_GREP_TIMEOUT_SEC)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return ToolResult(
            content=(
                f"grep timed out after {_GREP_TIMEOUT_SEC:.0f}s "
                "(pattern may cause catastrophic regex backtracking); simplify it"
            ),
            is_error=True,
        )
    if q.empty():
        return ToolResult(content="grep failed: no result from worker", is_error=True)
    status, *payload = q.get_nowait()
    if status == "err":
        return ToolResult(content=f"grep failed: {payload[0]}", is_error=True)
    hits, total = payload
    extra = (
        f"\n… truncated: showing {_GREP_MAX} of {total} matches …"
        if total > _GREP_MAX
        else ""
    )
    return ToolResult(content="\n".join(hits) + extra if hits else "(no matches)")


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write `content` via a sibling temp file + rename.

    A plain `write_text()` truncates the target before writing, so a failure
    mid-write (disk full, killed process) leaves a corrupted file. Writing to
    a temp file first and renaming it into place keeps the replace atomic on
    POSIX and Windows (`os.replace`), and `path` unchanged on any failure.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        try:
            shutil.copymode(path, tmp_name)
        except OSError:
            pass
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _edit(
    *,
    ctx: ToolCallContext,
    file_path: str,
    old_string: str,
    new_string: str,
    **_: Any,
) -> ToolResult:
    try:
        fp = _resolve_write_target(ctx, file_path)
    except ValueError as e:
        return ToolResult(content=str(e), is_error=True)
    if not fp.exists():
        return ToolResult(content=f"file not found: {fp}", is_error=True)
    if not fp.is_file():
        return ToolResult(content=f"not a file: {fp}", is_error=True)
    try:
        text = fp.read_text(encoding="utf-8")
    except OSError as e:
        return ToolResult(content=f"failed to read {fp}: {e}", is_error=True)
    count = text.count(old_string)
    if count == 0:
        return ToolResult(content=f"old_string not found in {fp}", is_error=True)
    if count > 1:
        return ToolResult(
            content=f"old_string is not unique in {fp} ({count} occurrences); refusing ambiguous edit",
            is_error=True,
        )
    try:
        _atomic_write_text(fp, text.replace(old_string, new_string, 1))
    except OSError as e:
        return ToolResult(content=f"failed to write {fp}: {e}", is_error=True)
    return ToolResult(content="ok")


def _resolve_shell_workdir(
    ctx: ToolCallContext,
    workdir: str | None,
    cwd: str | None,
) -> Path:
    raw = workdir or cwd
    if raw is None:
        return ctx.cwd
    return resolve_within_cwd(ctx.cwd, raw)


def _resolve_write_target(ctx: ToolCallContext, file_path: str) -> Path:
    """Resolve `file_path` against cwd, rejecting any target outside of it.

    `Write` runs with no OS-level sandbox (unlike apply_patch/Bash), so this
    boundary check is the only thing standing between the model and the rest
    of the filesystem.
    """
    return resolve_within_cwd(ctx.cwd, file_path)


def _write(
    *, ctx: ToolCallContext, file_path: str, content: str, **_: Any
) -> ToolResult:
    try:
        fp = _resolve_write_target(ctx, file_path)
    except ValueError as e:
        return ToolResult(content=str(e), is_error=True)
    existed = fp.exists()
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return ToolResult(
            content=f"failed to create parent directory {fp.parent}: {e}", is_error=True
        )
    try:
        _atomic_write_text(fp, content)
    except OSError as e:
        return ToolResult(content=f"failed to write {fp}: {e}", is_error=True)
    return ToolResult(content="overwritten" if existed else "created")


def _shell_command(
    *,
    ctx: ToolCallContext,
    cmd: str | None = None,
    command: str | None = None,
    workdir: str | None = None,
    cwd: str | None = None,
    timeout_sec: int | None = None,
    timeout_ms: int | None = None,
    login: bool = False,
    sandbox_permissions: str | None = None,
    sandbox_policy: str = "off",
    dangerously_disable_sandbox: bool = False,
    **_: Any,
) -> ToolResult:
    shell_cmd = cmd or command or ""
    if not shell_cmd:
        return ToolResult(content="missing cmd/command", is_error=True)
    try:
        run_cwd = str(_resolve_shell_workdir(ctx, workdir, cwd))
    except ValueError as e:
        return ToolResult(content=str(e), is_error=True)
    args = {
        "timeout_ms": timeout_ms,
        "timeout_sec": timeout_sec,
        "sandbox_permissions": sandbox_permissions,
        "dangerously_disable_sandbox": dangerously_disable_sandbox,
    }
    timeout = shell_timeout_ms(args) / 1000.0
    policy = _effective_shell_policy(ctx, args, sandbox_policy)

    argv, extra_env, label = build_bash_exec(
        shell_cmd,
        cwd=run_cwd,
        policy=normalize_policy(policy),
        dangerously_disable_sandbox=dangerously_disable_sandbox,
        timeout=int(timeout),
        login=login,
    )

    kind, parsed = maybe_parse_apply_patch(argv)
    if kind is MaybeApplyPatch.BODY and parsed is not None:
        try:
            return ToolResult(content=apply_parsed_patch(parsed, Path(run_cwd)))
        except (OSError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)
    if kind is MaybeApplyPatch.IMPLICIT:
        return ToolResult(
            content="patch detected without explicit apply_patch invocation; use apply_patch tool",
            is_error=True,
        )

    run_kw: dict[str, Any] = {
        "args": argv,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "cwd": run_cwd,
    }
    if extra_env is not None:
        run_kw["env"] = extra_env
    try:
        proc = subprocess.run(**run_kw)
    except subprocess.TimeoutExpired:
        ms = int(timeout * 1000)
        return ToolResult(content=f"timed out after {ms}ms", is_error=True)
    except OSError as e:
        return ToolResult(content=str(e), is_error=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > SHELL_MAX_BYTES:
        out = out[:SHELL_MAX_BYTES] + "\n… truncated …"
    tail = f"\nexit={proc.returncode}\n[{label}]"
    return ToolResult(content=out + tail, is_error=proc.returncode != 0)


def _exec_command(*, ctx: ToolCallContext, **kwargs: Any) -> ToolResult:
    return ctx.exec.exec_command(ctx, kwargs)


def _write_stdin(*, ctx: ToolCallContext, **kwargs: Any) -> ToolResult:
    return ctx.exec.write_stdin(ctx, kwargs)


def _effective_shell_policy(
    ctx: ToolCallContext, args: dict[str, Any], fallback: str
) -> str:
    if args.get("dangerously_disable_sandbox", ctx.dangerously_disable_sandbox):
        return "off"
    perm = args.get("sandbox_permissions")
    if perm == "require_escalated":
        return "off"
    if perm == "with_additional_permissions":
        return "restricted"
    return fallback or ctx.sandbox_policy


def _apply_patch(
    *,
    ctx: ToolCallContext,
    patch: str | None = None,
    input: str | None = None,
    command: list[str] | None = None,
    **kwargs: Any,
) -> ToolResult:
    if command:
        argv = [str(x) for x in command]
        kind, parsed = maybe_parse_apply_patch(argv)
        if kind is MaybeApplyPatch.BODY and parsed is not None:
            try:
                return ToolResult(content=apply_parsed_patch(parsed, ctx.cwd))
            except (OSError, ValueError) as e:
                return ToolResult(content=str(e), is_error=True)
        if kind is MaybeApplyPatch.IMPLICIT:
            return ToolResult(
                content="patch detected without explicit apply_patch invocation; use apply_patch tool",
                is_error=True,
            )
        return ToolResult(content="not an apply_patch command", is_error=True)

    raw = patch or input or kwargs.get("arguments")
    if isinstance(raw, dict):
        raw = raw.get("patch") or raw.get("input")
    if not raw or not isinstance(raw, str):
        return ToolResult(content="apply_patch expects a patch string", is_error=True)
    try:
        summary = apply_parsed_patch(ParsedPatch(patch=raw), ctx.cwd)
    except (OSError, ValueError) as e:
        return ToolResult(content=str(e), is_error=True)
    return ToolResult(content=summary)


def _resolve_view_image_target(ctx: ToolCallContext, path: str) -> Path:
    """Resolve `path` against cwd, rejecting any target outside of it."""
    target = _resolve_path(ctx, path).resolve()
    root = ctx.cwd
    if target != root and root not in target.parents:
        raise ValueError(
            f"refusing to read outside working directory: {target} (cwd: {root})"
        )
    return target


def _sniff_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _view_image(
    *, ctx: ToolCallContext, path: str, detail: str = "high", **_: Any
) -> ToolResult:
    if detail not in {"high", "original"}:
        return ToolResult(
            content=(
                f"view_image.detail only supports `high` or `original`; omit `detail` for default "
                f"high resized behavior, got `{detail}`"
            ),
            is_error=True,
        )
    try:
        abs_path = _resolve_view_image_target(ctx, path)
    except ValueError as e:
        return ToolResult(content=str(e), is_error=True)
    if not abs_path.is_file():
        return ToolResult(
            content=f"image path `{abs_path}` is not a file", is_error=True
        )
    try:
        data = abs_path.read_bytes()
    except OSError as e:
        return ToolResult(
            content=f"unable to read image at `{abs_path}`: {e}", is_error=True
        )
    if len(data) > _VIEW_IMAGE_CAP:
        return ToolResult(
            content=(
                f"Image too large for view_image: {len(data)} bytes exceeds the "
                f"{_VIEW_IMAGE_CAP} byte cap."
            ),
            is_error=True,
        )

    mime = _sniff_image_mime(data)
    if mime is None:
        return ToolResult(
            content="not a recognized image format (png/jpeg/gif/webp)",
            is_error=True,
        )
    out_bytes = data
    if detail == "high":
        out_bytes, mime = _maybe_resize_image(data, mime)

    b64 = base64.standard_b64encode(out_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    structured = {
        "content_items": [
            {"type": "input_image", "image_url": data_url, "detail": detail},
        ],
        "path": str(abs_path),
        "bytes": len(out_bytes),
    }
    return ToolResult(
        content=f"Attached image {abs_path} (detail={detail}, {len(out_bytes)} bytes)",
        structured=structured,
    )


def _maybe_resize_image(data: bytes, mime: str) -> tuple[bytes, str]:
    try:
        from PIL import Image
        import io
    except ImportError:
        return data, mime
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if max(w, h) <= _HIGH_DETAIL_MAX_PX:
        return data, mime
    scale = _HIGH_DETAIL_MAX_PX / max(w, h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    fmt = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/gif": "GIF",
        "image/webp": "WEBP",
    }.get(mime, "PNG")
    resized.save(buf, format=fmt)
    return buf.getvalue(), mime


def _update_plan(*, ctx: ToolCallContext, **raw: Any) -> ToolResult:
    # dispatch_tool injects cwd/sandbox fields; only validate model-visible args.
    injected = frozenset({"cwd", "sandbox_policy", "dangerously_disable_sandbox"})
    tool_args = {k: v for k, v in raw.items() if k not in injected}
    try:
        args = UpdatePlanArgs.from_dict(tool_args)
    except (TypeError, ValueError) as e:
        return ToolResult(
            content=f"failed to parse update_plan arguments: {e}", is_error=True
        )
    ctx.session_nonnull.update_plan(args)
    return ToolResult(content=PLAN_UPDATED)


def _new_context(*, ctx: ToolCallContext, **_: Any) -> ToolResult:
    try:
        ctx.session_nonnull.request_new_context()
    except RuntimeError as e:
        return ToolResult(content=str(e), is_error=True)
    return ToolResult(content=NEW_CONTEXT_MESSAGE)


def _get_context_remaining(*, ctx: ToolCallContext, **_: Any) -> ToolResult:
    remaining = ctx.session_nonnull.tokens_remaining()
    payload = (
        {"tokens_remaining": remaining, "status": "ok"}
        if remaining is not None
        else {"tokens_remaining": None, "status": "unknown"}
    )
    return ToolResult(content=json.dumps(payload, indent=2))


def _request_user_input(*, ctx: ToolCallContext, **raw: Any) -> ToolResult:
    try:
        validated = validate_request_user_input(raw)
        response = ctx.session_nonnull.request_user_input(args_to_payload(validated))
    except (RuntimeError, ValueError) as e:
        return ToolResult(content=str(e), is_error=True)
    return ToolResult(content=json.dumps(response, indent=2))


def _tool_search(
    *,
    ctx: ToolCallContext,
    query: str | None = None,
    limit: int | None = None,
    **_: Any,
) -> ToolResult:
    q = (query or "").strip()
    if not q:
        return ToolResult(content="tool_search requires non-empty query", is_error=True)
    q = q[:TOOL_SEARCH_MAX_QUERY_CHARS]
    hits = ctx.tool_catalog.search(q, normalize_tool_search_limit(limit))
    payload = {"tools": [h.to_loadable_json() for h in hits]}
    return ToolResult(content=json.dumps(payload, indent=2))


def _list_available_plugins(*, ctx: ToolCallContext, **_: Any) -> ToolResult:
    try:
        ctx.tool_catalog.refresh_from_codex()
    except ToolCatalogRefreshError as e:
        return ToolResult(content=str(e), is_error=True)
    payload = {"tools": ctx.tool_catalog.list_installable_entries()}
    return ToolResult(content=json.dumps(payload, indent=2))


def _request_plugin_install(*, ctx: ToolCallContext, **raw: Any) -> ToolResult:
    tool_id = str(raw.get("tool_id") or raw.get("plugin_id") or "").strip()
    if not tool_id:
        return ToolResult(
            content="request_plugin_install requires tool_id", is_error=True
        )
    tool_type_raw = str(raw.get("tool_type") or "plugin")
    action_type_raw = str(raw.get("action_type") or "install")
    suggest_reason = str(raw.get("suggest_reason") or raw.get("reason") or "").strip()
    if not suggest_reason:
        return ToolResult(
            content="request_plugin_install requires non-empty suggest_reason",
            is_error=True,
        )
    if action_type_raw != DiscoverableToolAction.INSTALL.value:
        return ToolResult(
            content='plugin install requests currently support only action_type="install"',
            is_error=True,
        )
    try:
        tool_type = DiscoverableToolType(tool_type_raw)
        action_type = DiscoverableToolAction(action_type_raw)
    except ValueError:
        return ToolResult(
            content=f"invalid tool_type or action_type: {tool_type_raw!r}, {action_type_raw!r}",
            is_error=True,
        )

    try:
        ctx.tool_catalog.refresh_from_codex()
    except ToolCatalogRefreshError as e:
        return ToolResult(content=str(e), is_error=True)
    tool = ctx.tool_catalog.find_discoverable(tool_type, tool_id)
    if tool is None:
        return ToolResult(content=f"unknown plugin {tool_id!r}", is_error=True)

    elicitation: dict[str, Any] = {
        "tool_type": tool_type.value,
        "action_type": action_type.value,
        "tool_id": tool_id,
        "tool_name": tool.name,
        "suggest_reason": suggest_reason,
    }
    if isinstance(tool, DiscoverablePlugin):
        elicitation["meta"] = build_plugin_install_elicitation_meta(
            tool, suggest_reason=suggest_reason, action_type=action_type
        )

    try:
        confirmed = ctx.session_nonnull.request_plugin_install(elicitation)
    except RuntimeError as e:
        return ToolResult(content=str(e), is_error=True)
    if confirmed is not True:
        confirmed = False

    try:
        outcome = execute_plugin_install(
            tool_type=tool_type,
            action_type=action_type,
            tool_id=tool_id,
            suggest_reason=suggest_reason,
            user_confirmed=confirmed,
        )
    except ValueError as e:
        return ToolResult(content=str(e), is_error=True)

    if outcome.deferred_tools:
        for entry in outcome.deferred_tools:
            ctx.tool_catalog.register_deferred(entry)
    ctx.tool_catalog.refresh_from_codex()
    return ToolResult(content=json.dumps(outcome.to_result_json(), indent=2))


def _exec(*, ctx: ToolCallContext, **kwargs: Any) -> ToolResult:
    from pulsing.forge.code_mode.handlers import handle_exec

    return handle_exec(ctx=ctx, **kwargs)


def _wait(*, ctx: ToolCallContext, **kwargs: Any) -> ToolResult:
    from pulsing.forge.code_mode.handlers import handle_wait

    return handle_wait(ctx=ctx, **kwargs)
