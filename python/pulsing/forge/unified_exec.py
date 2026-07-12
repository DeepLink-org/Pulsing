# SPDX-License-Identifier: Apache-2.0
"""UnifiedExec sessions for exec_command / write_stdin."""

from __future__ import annotations

import json
import os
import pty
import select
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pulsing.forge.context import ToolCallContext, resolve_within_cwd
from pulsing.forge.exec_output import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_STDIN_BYTES,
    ExecCommandOutput,
    ExecOutputDelta,
    ExecStream,
    OutputBuffer,
    Utf8ChunkDecoder,
    clamp_yield_ms,
)
from pulsing.forge.result import ToolResult
from pulsing.forge.sandbox import build_bash_exec, normalize_policy


@dataclass
class _PtyHandle:
    master_fd: int
    proc: subprocess.Popen[str]

    def write_stdin(self, data: bytes) -> None:
        os.write(self.master_fd, data)

    def kill(self) -> None:
        self.proc.kill()

    def poll(self) -> int | None:
        return self.proc.poll()

    def close(self) -> None:
        try:
            os.close(self.master_fd)
        except OSError:
            pass


@dataclass
class _ExecSession:
    proc: subprocess.Popen[str] | None
    pty: _PtyHandle | None
    buffer: OutputBuffer
    started: float
    tty: bool


class UnifiedExecManager:
    def __init__(self) -> None:
        self._next_id = 1
        self._sessions: dict[int, _ExecSession] = {}
        self._lock = threading.Lock()

    def __del__(self) -> None:
        try:
            self.stop_all()
        except Exception:
            pass

    def exec_command(self, ctx: ToolCallContext, args: dict[str, Any]) -> ToolResult:
        cmd = args.get("cmd") or args.get("command")
        if not cmd:
            return ToolResult(content="missing cmd/command", is_error=True)
        try:
            workdir = _resolve_workdir(ctx, args)
        except ValueError as e:
            return ToolResult(content=str(e), is_error=True)
        tty = bool(args.get("tty", True))
        login = bool(args.get("login", False))
        yield_ms = clamp_yield_ms(args.get("yield_time_ms"))
        max_tokens = int(args.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)
        policy = _effective_policy(ctx, args)
        if tty and normalize_policy(policy) != "off":
            return ToolResult(
                content=(
                    "tty exec_command sessions cannot use sandbox policy; "
                    "set tty=false or sandbox_permissions=require_escalated"
                ),
                is_error=True,
            )

        argv, env, _label = build_bash_exec(
            str(cmd),
            cwd=str(workdir),
            policy=normalize_policy(policy),
            dangerously_disable_sandbox=bool(
                args.get("dangerously_disable_sandbox", ctx.dangerously_disable_sandbox)
            ),
            timeout=120,
            login=login,
        )

        buffer = OutputBuffer()
        session_id = self._next_id
        self._next_id += 1
        on_delta = _stream_hook(ctx, session_id)

        if tty:
            pty_handle = _spawn_pty(argv, workdir, env, buffer, session_id, on_delta)
            with self._lock:
                self._sessions[session_id] = _ExecSession(
                    proc=None,
                    pty=pty_handle,
                    buffer=buffer,
                    started=time.time(),
                    tty=True,
                )
        else:
            proc = _spawn_pipe(argv, workdir, env, buffer, session_id, on_delta)
            with self._lock:
                self._sessions[session_id] = _ExecSession(
                    proc=proc,
                    pty=None,
                    buffer=buffer,
                    started=time.time(),
                    tty=False,
                )

        time.sleep(yield_ms / 1000.0)
        return self._poll(session_id, max_tokens)

    def write_stdin(self, ctx: ToolCallContext, args: dict[str, Any]) -> ToolResult:
        _ = ctx
        raw_session_id = args.get("session_id")
        if raw_session_id is None:
            return ToolResult(content="missing session_id", is_error=True)
        try:
            session_id = int(raw_session_id)
        except (TypeError, ValueError):
            return ToolResult(
                content=f"invalid session_id {raw_session_id!r}", is_error=True
            )
        chars = args["chars"] if "chars" in args else args.get("input")
        if chars is None:
            return ToolResult(content="missing chars", is_error=True)
        chars = str(chars)
        char_bytes = len(chars.encode("utf-8", errors="surrogateescape"))
        if char_bytes > MAX_STDIN_BYTES:
            return ToolResult(
                content=(
                    f"stdin input too large: {char_bytes} bytes (max {MAX_STDIN_BYTES})"
                ),
                is_error=True,
            )
        yield_ms = clamp_yield_ms(args.get("yield_time_ms"))
        max_tokens = int(args.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)

        # Hold the lock across the actual write so it can't race a concurrent
        # `_poll` that closes/pops this session out from under us.
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return ToolResult(
                    content=f"unknown session_id {session_id}", is_error=True
                )
            if chars != "\x03":
                exit_code = (
                    session.pty.poll()
                    if session.pty is not None
                    else session.proc.poll() if session.proc is not None else None
                )
                if exit_code is not None:
                    return ToolResult(
                        content=f"session {session_id} has already exited",
                        is_error=True,
                    )
            try:
                if chars == "\x03":
                    if session.pty is not None:
                        session.pty.kill()
                    elif session.proc is not None:
                        session.proc.kill()
                elif not session.tty:
                    return ToolResult(
                        content="stdin writes require tty=true exec_command sessions",
                        is_error=True,
                    )
                elif session.pty is not None:
                    session.pty.write_stdin(
                        chars.encode("utf-8", errors="surrogateescape")
                    )
                elif session.proc is not None and session.proc.stdin is not None:
                    session.proc.stdin.write(chars)
                    session.proc.stdin.flush()
                else:
                    return ToolResult(content="session stdin is closed", is_error=True)
            except (OSError, ValueError) as e:
                return ToolResult(content=f"write stdin failed: {e}", is_error=True)

        time.sleep(yield_ms / 1000.0)
        return self._poll(session_id, max_tokens)

    def stop_all(self) -> int:
        """Kill every live session (host/agent teardown — avoid orphaned children)."""
        with self._lock:
            sessions = self._sessions
            self._sessions = {}
        for session in sessions.values():
            proc = session.pty.proc if session.pty is not None else session.proc
            try:
                if session.pty is not None:
                    session.pty.kill()
                    session.pty.close()
                elif proc is not None:
                    proc.kill()
                if proc is not None:
                    proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
        return len(sessions)

    def _poll(self, session_id: int, max_tokens: int) -> ToolResult:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return ToolResult(content=f"unknown session_id {session_id}", is_error=True)

        if session.pty is not None:
            exit_code = session.pty.poll()
        else:
            assert session.proc is not None
            exit_code = session.proc.poll()

        wall = time.time() - session.started
        session.buffer.truncate_to_tokens(max_tokens)
        output = session.buffer.snapshot()
        structured = ExecCommandOutput.build(
            output=output,
            wall_time_seconds=wall,
            exit_code=exit_code,
            session_id=None if exit_code is not None else session_id,
        )
        if exit_code is not None:
            with self._lock:
                closed = self._sessions.pop(session_id, None)
            if closed and closed.pty is not None:
                closed.pty.close()
        payload = structured.to_dict()
        return ToolResult(
            content=json.dumps(payload, indent=2),
            is_error=exit_code is not None and exit_code != 0,
            structured=payload,
        )


def _stream_hook(
    ctx: ToolCallContext,
    session_id: int,
) -> Callable[[ExecOutputDelta], None] | None:
    session = ctx.session
    if session is None:
        return None

    def emit(delta: ExecOutputDelta) -> None:
        session.on_exec_output_delta(delta)

    return emit


def _emit_delta(
    on_delta: Callable[[ExecOutputDelta], None] | None,
    session_id: int,
    stream: ExecStream,
    chunk: str,
) -> None:
    if not chunk or on_delta is None:
        return
    on_delta(ExecOutputDelta(session_id=session_id, stream=stream, chunk=chunk))


def _spawn_pty(
    argv: list[str],
    workdir: Path,
    env: dict[str, str] | None,
    buffer: OutputBuffer,
    session_id: int,
    on_delta: Callable[[ExecOutputDelta], None] | None,
) -> _PtyHandle:
    master_fd, slave_fd = pty.openpty()
    popen_kw: dict[str, Any] = {
        "args": argv,
        "cwd": str(workdir),
        "stdin": slave_fd,
        "stdout": slave_fd,
        "stderr": slave_fd,
        "text": True,
        "close_fds": True,
    }
    if env is not None:
        popen_kw["env"] = env
    proc = subprocess.Popen(**popen_kw)
    os.close(slave_fd)

    def _reader() -> None:
        decoder = Utf8ChunkDecoder()
        while True:
            if proc.poll() is not None:
                break
            try:
                ready, _, _ = select.select([master_fd], [], [], 0.05)
            except OSError:
                break
            if not ready:
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                buffer.push(text)
                _emit_delta(on_delta, session_id, ExecStream.PTY, text)
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                buffer.push(text)
                _emit_delta(on_delta, session_id, ExecStream.PTY, text)
        tail = decoder.finish()
        if tail:
            buffer.push(tail)
            _emit_delta(on_delta, session_id, ExecStream.PTY, tail)

    threading.Thread(target=_reader, daemon=True).start()
    return _PtyHandle(master_fd=master_fd, proc=proc)


def _spawn_pipe(
    argv: list[str],
    workdir: Path,
    env: dict[str, str] | None,
    buffer: OutputBuffer,
    session_id: int,
    on_delta: Callable[[ExecOutputDelta], None] | None,
) -> subprocess.Popen[str]:
    popen_kw: dict[str, Any] = {
        "args": argv,
        "cwd": str(workdir),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "bufsize": 1,
    }
    if env is not None:
        popen_kw["env"] = env
    proc = subprocess.Popen(**popen_kw)

    def _reader() -> None:
        assert proc.stdout is not None
        assert proc.stderr is not None
        while proc.poll() is None:
            if proc.stdout.readable():
                chunk = proc.stdout.read(4096)
                if chunk:
                    buffer.push(chunk)
                    _emit_delta(on_delta, session_id, ExecStream.STDOUT, chunk)
            if proc.stderr.readable():
                chunk = proc.stderr.read(4096)
                if chunk:
                    buffer.push(chunk)
                    _emit_delta(on_delta, session_id, ExecStream.STDERR, chunk)
            time.sleep(0.01)
        rest_out = proc.stdout.read() or ""
        rest_err = proc.stderr.read() or ""
        if rest_out:
            buffer.push(rest_out)
            _emit_delta(on_delta, session_id, ExecStream.STDOUT, rest_out)
        if rest_err:
            buffer.push(rest_err)
            _emit_delta(on_delta, session_id, ExecStream.STDERR, rest_err)

    threading.Thread(target=_reader, daemon=True).start()
    return proc


def _resolve_workdir(ctx: ToolCallContext, args: dict[str, Any]) -> Path:
    raw = args.get("workdir") or args.get("cwd")
    if raw is None:
        return ctx.cwd
    return resolve_within_cwd(ctx.cwd, str(raw))


def _effective_policy(ctx: ToolCallContext, args: dict[str, Any]) -> str:
    if args.get("dangerously_disable_sandbox", ctx.dangerously_disable_sandbox):
        return "off"
    perm = args.get("sandbox_permissions")
    if perm == "require_escalated":
        return "off"
    if perm == "with_additional_permissions":
        return "restricted"
    return ctx.sandbox_policy
