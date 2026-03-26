"""ProcessActor — wraps a subprocess.Popen instance as a distributed actor."""

from __future__ import annotations

import asyncio
import subprocess

import pulsing as pul


@pul.remote
class ProcessActor:
    """Actor that manages a subprocess.Popen instance.

    Top-level methods (poll, wait, communicate, send_signal, terminate, kill)
    match subprocess.Popen exactly.

    stdin/stdout/stderr are file objects in subprocess.Popen; actors cannot
    pass file objects across the network, so their operations are exposed as
    prefixed methods:  stdin_write, stdin_close, stdout_read, stdout_readline,
    stderr_read, stderr_readline.  The popen.py wrapper re-assembles these
    into .stdin / .stdout / .stderr proxy objects so user code looks identical
    to stdlib subprocess.
    """

    def __init__(
        self,
        args,
        *,
        stdin=None,
        stdout=None,
        stderr=None,
        cwd=None,
        env=None,
        shell=False,
        encoding=None,
        errors=None,
        text=False,
        bufsize=-1,
    ):
        self._proc = subprocess.Popen(
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
            bufsize=bufsize,
        )

    # ------------------------------------------------------------------
    # Popen attributes (exposed as methods; actors can't have properties
    # that cross the network boundary)
    # ------------------------------------------------------------------

    def pid(self) -> int:
        return self._proc.pid

    def returncode(self) -> int | None:
        return self._proc.returncode

    # ------------------------------------------------------------------
    # subprocess.Popen methods — identical name and signature
    # ------------------------------------------------------------------

    def poll(self) -> int | None:
        return self._proc.poll()

    async def wait(self, timeout: float | None = None) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._proc.wait(timeout=timeout)
        )

    async def communicate(
        self, input: bytes | str | None = None, timeout: float | None = None
    ) -> tuple:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._proc.communicate(input=input, timeout=timeout)
        )

    def send_signal(self, sig: int) -> None:
        self._proc.send_signal(sig)

    def terminate(self) -> None:
        self._proc.terminate()

    def kill(self) -> None:
        self._proc.kill()

    # ------------------------------------------------------------------
    # stdin methods  (mirror proc.stdin.write / flush / close)
    # ------------------------------------------------------------------

    def stdin_write(self, data: bytes | str) -> None:
        if self._proc.stdin is None:
            raise OSError("stdin is not PIPE")
        if isinstance(data, str):
            data = data.encode()
        self._proc.stdin.write(data)

    def stdin_flush(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.flush()

    def stdin_close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()

    def stdin_fileno(self) -> int | None:
        if self._proc.stdin is None:
            return None
        return self._proc.stdin.fileno()

    # ------------------------------------------------------------------
    # stdout methods  (mirror proc.stdout.read / readline / readlines / close)
    # ------------------------------------------------------------------

    async def stdout_read(self, n: int = -1) -> bytes | None:
        if self._proc.stdout is None:
            return None
        loop = asyncio.get_running_loop()
        if n == -1:
            return await loop.run_in_executor(None, self._proc.stdout.read)
        return await loop.run_in_executor(None, self._proc.stdout.read, n)

    async def stdout_readline(self) -> bytes | None:
        if self._proc.stdout is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._proc.stdout.readline)

    async def stdout_readlines(self) -> list | None:
        if self._proc.stdout is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._proc.stdout.readlines)

    def stdout_close(self) -> None:
        if self._proc.stdout is not None:
            self._proc.stdout.close()

    def stdout_fileno(self) -> int | None:
        if self._proc.stdout is None:
            return None
        return self._proc.stdout.fileno()

    # ------------------------------------------------------------------
    # stderr methods  (mirror proc.stderr.read / readline / readlines / close)
    # ------------------------------------------------------------------

    async def stderr_read(self, n: int = -1) -> bytes | None:
        if self._proc.stderr is None:
            return None
        loop = asyncio.get_running_loop()
        if n == -1:
            return await loop.run_in_executor(None, self._proc.stderr.read)
        return await loop.run_in_executor(None, self._proc.stderr.read, n)

    async def stderr_readline(self) -> bytes | None:
        if self._proc.stderr is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._proc.stderr.readline)

    async def stderr_readlines(self) -> list | None:
        if self._proc.stderr is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._proc.stderr.readlines)

    def stderr_close(self) -> None:
        if self._proc.stderr is not None:
            self._proc.stderr.close()

    def stderr_fileno(self) -> int | None:
        if self._proc.stderr is None:
            return None
        return self._proc.stderr.fileno()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_stop(self):
        if self._proc.poll() is None:
            try:
                self._proc.kill()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# stdin / stdout / stderr proxy objects
#
# subprocess.Popen exposes .stdin / .stdout / .stderr as file-like objects.
# These proxies replicate that interface, forwarding each call to the
# corresponding actor method via an injected run_sync function.
#
#     proc.stdin.write(b"hello")
#     proc.stdin.close()
#     line = proc.stdout.readline()
# ---------------------------------------------------------------------------


class _StdinProxy:
    """Mirrors proc.stdin: write / flush / close / fileno."""

    def __init__(self, actor_proxy, run_sync):
        self._proxy = actor_proxy
        self._run_sync = run_sync

    def write(self, data: bytes | str) -> None:
        self._run_sync(self._proxy.stdin_write(data))

    def flush(self) -> None:
        self._run_sync(self._proxy.stdin_flush())

    def close(self) -> None:
        self._run_sync(self._proxy.stdin_close())

    def fileno(self) -> int | None:
        return self._run_sync(self._proxy.stdin_fileno())


class _StdoutProxy:
    """Mirrors proc.stdout: read / readline / readlines / close / fileno."""

    def __init__(self, actor_proxy, run_sync):
        self._proxy = actor_proxy
        self._run_sync = run_sync

    def read(self, n: int = -1) -> bytes | None:
        return self._run_sync(self._proxy.stdout_read(n))

    def readline(self) -> bytes | None:
        return self._run_sync(self._proxy.stdout_readline())

    def readlines(self) -> list | None:
        return self._run_sync(self._proxy.stdout_readlines())

    def close(self) -> None:
        self._run_sync(self._proxy.stdout_close())

    def fileno(self) -> int | None:
        return self._run_sync(self._proxy.stdout_fileno())


class _StderrProxy:
    """Mirrors proc.stderr: read / readline / readlines / close / fileno."""

    def __init__(self, actor_proxy, run_sync):
        self._proxy = actor_proxy
        self._run_sync = run_sync

    def read(self, n: int = -1) -> bytes | None:
        return self._run_sync(self._proxy.stderr_read(n))

    def readline(self) -> bytes | None:
        return self._run_sync(self._proxy.stderr_readline())

    def readlines(self) -> list | None:
        return self._run_sync(self._proxy.stderr_readlines())

    def close(self) -> None:
        self._run_sync(self._proxy.stderr_close())

    def fileno(self) -> int | None:
        return self._run_sync(self._proxy.stderr_fileno())
