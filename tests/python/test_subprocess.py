"""Tests for pulsing.subprocess module."""

import asyncio
import pytest
import subprocess as _subprocess

import pulsing as pul
from pulsing.subprocess import (
    PIPE,
    Popen,
    CompletedProcess,
    run,
    call,
    check_call,
    check_output,
)


@pytest.fixture(autouse=True)
async def pulsing_system():
    await pul.init()
    yield
    await pul.shutdown()


def _in_executor(fn, *args):
    """Run a sync pulsing.subprocess call in a thread-pool worker.

    pulsing.subprocess functions block via run_coroutine_threadsafe and must
    not be called from the event loop thread directly.
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, fn, *args)


# ---------------------------------------------------------------------------
# Popen basic functionality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_popen_pid():
    def _test():
        proc = Popen(["echo", "hello"], stdout=PIPE)
        proc.communicate()
        assert isinstance(proc.pid, int) and proc.pid > 0

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_communicate_captures_stdout():
    def _test():
        proc = Popen(["echo", "pulsing"], stdout=PIPE, stderr=PIPE)
        stdout, _ = proc.communicate()
        assert stdout.strip() == b"pulsing"
        assert proc.returncode == 0

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_nonzero_returncode():
    def _test():
        proc = Popen(["false"])
        proc.wait()
        assert proc.returncode != 0

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_poll():
    def _test():
        proc = Popen(["sleep", "0.05"])
        proc.wait()
        assert proc.poll() == 0

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_terminate():
    def _test():
        proc = Popen(["sleep", "60"])
        proc.terminate()
        assert proc.wait() is not None

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_kill():
    def _test():
        proc = Popen(["sleep", "60"])
        proc.kill()
        assert proc.wait() is not None

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_stdin_write():
    def _test():
        proc = Popen(["cat"], stdin=PIPE, stdout=PIPE)
        proc.stdin.write(b"hello stdin")
        proc.stdin.close()
        assert proc.stdout.read() == b"hello stdin"

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_context_manager():
    def _test():
        with Popen(["echo", "ctx"], stdout=PIPE) as proc:
            stdout, _ = proc.communicate()
        assert b"ctx" in stdout

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_read_stdout_lines():
    def _test():
        proc = Popen(["printf", "line1\\nline2\\n"], stdout=PIPE)
        assert proc.stdout.readline() == b"line1\n"
        assert proc.stdout.readline() == b"line2\n"
        proc.wait()

    await _in_executor(_test)


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_capture_output():
    def _test():
        result = run(["echo", "run works"], capture_output=True)
        assert result.returncode == 0
        assert b"run works" in result.stdout

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_run_check_raises():
    def _test():
        with pytest.raises(_subprocess.CalledProcessError):
            run(["false"], check=True)

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_call_returncode():
    def _test():
        assert call(["true"]) == 0
        assert call(["false"]) != 0

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_check_call_raises():
    def _test():
        with pytest.raises(_subprocess.CalledProcessError):
            check_call(["false"])

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_check_output_returns_bytes():
    def _test():
        out = check_output(["echo", "output_test"])
        assert b"output_test" in out

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_check_output_raises_on_error():
    def _test():
        with pytest.raises(_subprocess.CalledProcessError):
            check_output(["false"])

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_run_stdin_input():
    def _test():
        result = run(["cat"], input=b"piped input", capture_output=True)
        assert result.returncode == 0
        assert result.stdout == b"piped input"

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_run_shell_true():
    def _test():
        result = run("echo shell_mode", shell=True, capture_output=True)
        assert result.returncode == 0
        assert b"shell_mode" in result.stdout

    await _in_executor(_test)


def test_completed_process_check_returncode():
    with pytest.raises(_subprocess.CalledProcessError):
        CompletedProcess(args=["false"], returncode=1).check_returncode()
    CompletedProcess(args=["true"], returncode=0).check_returncode()
