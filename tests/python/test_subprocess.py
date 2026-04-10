"""Tests for pulsing.subprocess native and resource-backed behavior."""

import asyncio
import subprocess as _subprocess
import threading

import pytest

import pulsing as pul
from pulsing._async_bridge import clear_pulsing_loop, get_shared_loop
import pulsing._runtime as _rt
import pulsing.subprocess.popen as popen_module
from pulsing.subprocess import (
    PIPE,
    CompletedProcess,
    Popen,
    call,
    check_output,
    check_call,
    run,
)


def _assert_has_bound_addr() -> None:
    addr = str(pul.get_system().addr)
    host, sep, port = addr.rpartition(":")
    assert sep == ":"
    assert host
    assert port.isdigit()


def _require_ray():
    return pytest.importorskip("ray")


def _enable_pulsing_backend(monkeypatch, value: str = "1") -> None:
    monkeypatch.setenv("USE_POLSING_SUBPROCESS", value)


@pytest.fixture(autouse=True)
def clear_subprocess_env(monkeypatch):
    monkeypatch.delenv("USE_POLSING_SUBPROCESS", raising=False)


@pytest.fixture(autouse=True)
async def clean_subprocess_state():
    yield
    _rt.shutdown()
    if pul.is_initialized():
        await pul.shutdown()
    clear_pulsing_loop()
    try:
        ray = __import__("ray")
    except ImportError:
        return
    if ray.is_initialized():
        ray.shutdown()


def test_native_backend_is_default_and_ignores_pulsing_kwargs_when_env_disabled():
    assert not pul.is_initialized()

    result = run(
        ["echo", "native"],
        capture_output=True,
        placement="remote",
        system=object(),
        name="ignored",
        resources={"num_cpus": 1},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == b"native"
    assert not pul.is_initialized()
    assert get_shared_loop() is None


def test_empty_resources_stays_native_even_with_env_enabled(monkeypatch):
    _enable_pulsing_backend(monkeypatch)

    result = run(
        ["echo", "empty_resources"],
        capture_output=True,
        resources={},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == b"empty_resources"
    assert not pul.is_initialized()


def test_non_empty_resources_stays_native_with_falsy_env(monkeypatch):
    _enable_pulsing_backend(monkeypatch, "off")

    result = run(
        ["echo", "falsy_env"],
        capture_output=True,
        resources={"num_cpus": 1},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == b"falsy_env"
    assert not pul.is_initialized()


def test_native_popen_text_mode():
    proc = Popen(
        ["cat"],
        stdin=PIPE,
        stdout=PIPE,
        text=True,
        placement="remote",
        system=object(),
        name="ignored",
    )
    stdout, _ = proc.communicate(input="hello text")

    assert stdout == "hello text"
    assert proc.returncode == 0
    assert not pul.is_initialized()


def test_native_run_shell_true():
    result = run("echo shell_mode", shell=True, capture_output=True)
    assert result.returncode == 0
    assert b"shell_mode" in result.stdout


def test_native_timeout_raises_stdlib_error():
    with pytest.raises(_subprocess.TimeoutExpired):
        run(
            ["python3", "-c", "import time; time.sleep(1)"],
            timeout=0.01,
            capture_output=True,
        )


def test_native_helpers_match_stdlib_shape():
    assert call(["true"]) == 0
    assert call(["false"]) != 0

    with pytest.raises(_subprocess.CalledProcessError):
        check_call(["false"])

    out = check_output(["echo", "output_test"])
    assert b"output_test" in out


def test_completed_process_check_returncode():
    with pytest.raises(_subprocess.CalledProcessError):
        CompletedProcess(args=["false"], returncode=1).check_returncode()

    CompletedProcess(args=["true"], returncode=0).check_returncode()


def test_resources_backend_auto_inits(monkeypatch):
    _require_ray()
    _enable_pulsing_backend(monkeypatch, "YES")

    result = run(
        ["echo", "pulsing"],
        capture_output=True,
        resources={"num_cpus": 1},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == b"pulsing"
    assert pul.is_initialized()
    assert _rt.owns_system() is True
    assert get_shared_loop() is not None
    _assert_has_bound_addr()


def test_resources_backend_reuses_auto_initialized_system(monkeypatch):
    _require_ray()
    _enable_pulsing_backend(monkeypatch)

    first = run(["echo", "one"], capture_output=True, resources={"num_cpus": 1})
    shared_loop = get_shared_loop()
    system = pul.get_system()
    second = run(["echo", "two"], capture_output=True, resources={"num_cpus": 1})

    assert first.stdout.strip() == b"one"
    assert second.stdout.strip() == b"two"
    assert pul.get_system() is system
    assert get_shared_loop() is shared_loop


def test_resources_popen_text_mode(monkeypatch):
    _require_ray()
    _enable_pulsing_backend(monkeypatch)

    proc = Popen(
        ["cat"],
        stdin=PIPE,
        stdout=PIPE,
        text=True,
        resources={"num_cpus": 1},
    )
    stdout, _ = proc.communicate(input="hello pulsing")

    assert stdout == "hello pulsing"
    assert proc.returncode == 0


def test_resources_timeout_raises_stdlib_error(monkeypatch):
    _require_ray()
    _enable_pulsing_backend(monkeypatch)

    with pytest.raises(_subprocess.TimeoutExpired):
        run(
            ["python3", "-c", "import time; time.sleep(1)"],
            timeout=0.01,
            capture_output=True,
            resources={"num_cpus": 1},
        )


@pytest.mark.asyncio
async def test_sync_api_inside_async_before_init_raises_clear_error_for_resources(
    monkeypatch,
):
    _require_ray()
    _enable_pulsing_backend(monkeypatch)

    with pytest.raises(
        RuntimeError, match="sync auto-init cannot run on the same thread"
    ):
        run(
            ["echo", "inside_async"],
            capture_output=True,
            resources={"num_cpus": 1},
        )


@pytest.mark.asyncio
async def test_same_loop_explicit_init_raises_clear_error_for_resources(monkeypatch):
    _require_ray()
    _enable_pulsing_backend(monkeypatch)
    await pul.init()

    with pytest.raises(RuntimeError, match="cannot block on the same event loop"):
        run(["echo", "blocked"], capture_output=True, resources={"num_cpus": 1})


def test_reuses_explicitly_initialized_background_system_with_resources(monkeypatch):
    _require_ray()
    _enable_pulsing_backend(monkeypatch)
    ready = threading.Event()
    stop = threading.Event()

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def keep_system_alive():
            await pul.init(addr="127.0.0.1:0")
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)
            await pul.shutdown()

        loop.run_until_complete(keep_system_alive())
        loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)

    try:
        system = pul.get_system()
        result = run(["echo", "reused"], capture_output=True, resources={"num_cpus": 1})
        assert result.stdout.strip() == b"reused"
        assert pul.get_system() is system
        assert _rt.owns_system() is False
    finally:
        stop.set()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_existing_explicit_system_still_supports_async_calls():
    await pul.init()

    @pul.remote
    class Echo:
        async def ping(self, value: str) -> str:
            return value

    ref = await Echo.spawn(name="subprocess_echo")
    assert await ref.ping("ok") == "ok"
