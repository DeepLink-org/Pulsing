"""
pulsing.subprocess — sync Popen and high-level helpers mirroring subprocess.

Public interface is **synchronous**, identical to stdlib subprocess.

How to use from async code
--------------------------
pulsing.subprocess functions block the calling thread.  When called from
within an async context (e.g. inside main()), run them in a thread-pool
executor so the event loop stays free:

    async def main():
        await pul.init()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_demos)   # run_demos is sync
        await pul.shutdown()

This is the same pattern you would use for any blocking I/O in async code.
"""

from __future__ import annotations

import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pulsing.exceptions import PulsingActorError

from .process import ProcessActor, _StdinProxy, _StdoutProxy, _StderrProxy


# ---------------------------------------------------------------------------
# Loop registry + sync dispatch
# ---------------------------------------------------------------------------

_pulsing_loop: asyncio.AbstractEventLoop | None = None


def _set_pulsing_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _pulsing_loop
    _pulsing_loop = loop


def _get_pulsing_loop() -> asyncio.AbstractEventLoop:
    if _pulsing_loop is not None:
        return _pulsing_loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError(
            "pulsing actor system loop not found. "
            "Call pul.init() before using pulsing.subprocess."
        )


async def _ensure_coro(obj) -> Any:
    """Await obj (works for both native coroutines and awaitable objects like _AsyncMethodCall)."""
    return await obj


def _run_sync(awaitable) -> Any:
    """Submit an awaitable to pulsing's loop and block the current thread until done.

    MUST be called from a thread that is NOT the pulsing loop thread.
    Use loop.run_in_executor to ensure this from async code.
    """
    return asyncio.run_coroutine_threadsafe(
        _ensure_coro(awaitable), _get_pulsing_loop()
    ).result()


def _maybe_raise_timeout_expired(
    error: PulsingActorError, args: Any, timeout: float | None
) -> None:
    if timeout is None:
        raise error

    if "timed out after" not in str(error):
        raise error

    raise subprocess.TimeoutExpired(args, timeout) from error


# ---------------------------------------------------------------------------
# CompletedProcess  (mirrors subprocess.CompletedProcess)
# ---------------------------------------------------------------------------


@dataclass
class CompletedProcess:
    args: Any
    returncode: int
    stdout: bytes | str | None = None
    stderr: bytes | str | None = None

    def check_returncode(self):
        if self.returncode != 0:
            raise subprocess.CalledProcessError(
                self.returncode, self.args, self.stdout, self.stderr
            )


# ---------------------------------------------------------------------------
# Popen  (synchronous interface, mirrors subprocess.Popen)
# ---------------------------------------------------------------------------


class Popen:
    """Synchronous remote-capable Popen, interface identical to subprocess.Popen.

    After construction, .stdin / .stdout / .stderr are proxy objects that
    forward calls (write, read, readline, close …) to the remote actor,
    exactly like subprocess.Popen.

    Must be called from a non-loop thread (e.g. inside loop.run_in_executor).
    """

    def __init__(self, args, **kwargs):
        self._placement = kwargs.pop("placement", "local")
        self._system = kwargs.pop("system", None)
        self._name = kwargs.pop("name", None)
        self._resources = kwargs.pop("resources", None)
        self._args = args
        self._spawn_kwargs = kwargs
        self._proxy = None

        self.returncode: int | None = None
        self.pid: int | None = None
        self.stdin: _StdinProxy | None = None
        self.stdout: _StdoutProxy | None = None
        self.stderr: _StderrProxy | None = None

        _run_sync(self._aspawn())

    async def _aspawn(self) -> None:
        if self._resources is not None:
            from .ray_spawn import ray_spawn_process_actor
            from pulsing.core import get_system

            system = self._system or get_system()
            self._proxy = await ray_spawn_process_actor(
                system,
                self._args,
                actor_name=self._name,
                resources=self._resources,
                **self._spawn_kwargs,
            )
        else:
            self._proxy = await ProcessActor.spawn(
                self._args,
                system=self._system,
                name=self._name,
                placement=self._placement,
                **self._spawn_kwargs,
            )
        self.pid = await self._proxy.pid()
        self.stdin = _StdinProxy(self._proxy, _run_sync)
        self.stdout = _StdoutProxy(self._proxy, _run_sync)
        self.stderr = _StderrProxy(self._proxy, _run_sync)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.wait()
        self.close()

    def __del__(self):
        # Best effort cleanup.  If the event loop is already closed, there's not much we can do.
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Clean up Ray actor resources if this Popen used resources.

        No-op for local (non-Ray) processes.  Idempotent.
        """
        if getattr(self._proxy, "_ray_node_actor", None) is not None:
            from .ray_spawn import cleanup_ray_actor

            _run_sync(cleanup_ray_actor(self._proxy))

    def poll(self) -> int | None:
        rc = _run_sync(self._proxy.poll())
        self.returncode = rc
        return rc

    def wait(self, timeout: float | None = None) -> int:
        rc = _run_sync(self._proxy.wait(timeout))
        self.returncode = rc
        return rc

    def communicate(self, input=None, timeout: float | None = None) -> tuple:
        async def _do():
            stdout, stderr = await self._proxy.communicate(input, timeout)
            self.returncode = await self._proxy.returncode()
            return stdout, stderr

        try:
            return _run_sync(_do())
        except PulsingActorError as error:
            _maybe_raise_timeout_expired(error, self._args, timeout)

    def send_signal(self, sig: int) -> None:
        _run_sync(self._proxy.send_signal(sig))

    def terminate(self) -> None:
        _run_sync(self._proxy.terminate())

    def kill(self) -> None:
        _run_sync(self._proxy.kill())


# ---------------------------------------------------------------------------
# High-level helpers  (synchronous, matching subprocess)
# ---------------------------------------------------------------------------


def run(
    args,
    *,
    stdin=None,
    input=None,
    stdout=None,
    stderr=None,
    capture_output: bool = False,
    cwd=None,
    env=None,
    shell: bool = False,
    encoding=None,
    errors=None,
    text: bool = False,
    timeout: float | None = None,
    check: bool = False,
    placement: str | int = "local",
    system=None,
    name: str | None = None,
    resources: dict | None = None,
) -> CompletedProcess:
    """Run a command, wait for it, return CompletedProcess. Mirrors subprocess.run()."""
    if capture_output:
        if stdout is not None or stderr is not None:
            raise ValueError("capture_output is mutually exclusive with stdout/stderr")
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

    if input is not None:
        if stdin is not None:
            raise ValueError("stdin and input are mutually exclusive")
        stdin = subprocess.PIPE

    proc = Popen(
        args,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        env=env,
        shell=shell,
        encoding=encoding,
        errors=errors,
        text=text,
        placement=placement,
        system=system,
        name=name,
        resources=resources,
    )
    try:
        out, err = proc.communicate(input=input, timeout=timeout)
    finally:
        proc.close()
    result = CompletedProcess(
        args=args, returncode=proc.returncode, stdout=out, stderr=err
    )
    if check:
        result.check_returncode()
    return result


def call(
    args, *, timeout=None, placement="local", system=None, resources=None, **kwargs
) -> int:
    return run(
        args,
        timeout=timeout,
        placement=placement,
        system=system,
        resources=resources,
        **kwargs,
    ).returncode


def check_call(
    args, *, timeout=None, placement="local", system=None, resources=None, **kwargs
) -> int:
    return run(
        args,
        timeout=timeout,
        check=True,
        placement=placement,
        system=system,
        resources=resources,
        **kwargs,
    ).returncode


def check_output(
    args,
    *,
    stderr=None,
    timeout=None,
    placement="local",
    system=None,
    resources=None,
    **kwargs,
) -> bytes | str:
    return run(
        args,
        stdout=subprocess.PIPE,
        stderr=stderr,
        timeout=timeout,
        check=True,
        placement=placement,
        system=system,
        resources=resources,
        **kwargs,
    ).stdout
