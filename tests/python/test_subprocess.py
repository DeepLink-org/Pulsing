"""Tests for pulsing.subprocess native and resource-backed behavior."""

import asyncio
import subprocess as _subprocess
import threading
from types import SimpleNamespace

import pytest

import pulsing as pul
from pulsing._async_bridge import (
    clear_pulsing_loop,
    get_shared_loop,
    run_sync,
    stop_shared_loop,
)
import pulsing._runtime as _rt
import pulsing.subprocess.popen as popen_module
import pulsing.subprocess.process as process_module
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


async def _shutdown_ray_on_shared_loop() -> None:
    try:
        import ray
    except ImportError:
        return

    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(autouse=True)
def clear_subprocess_env(monkeypatch):
    monkeypatch.delenv("USE_POLSING_SUBPROCESS", raising=False)


@pytest.fixture(autouse=True)
async def clean_subprocess_state():
    yield
    if get_shared_loop() is not None:
        try:
            run_sync(_shutdown_ray_on_shared_loop(), timeout=30)
        except Exception:
            pass
    _rt.shutdown()
    if pul.is_initialized():
        await pul.shutdown()
    stop_shared_loop()
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
async def test_sync_api_inside_async_before_init_auto_inits_for_resources(
    monkeypatch,
):
    _require_ray()
    _enable_pulsing_backend(monkeypatch)

    result = run(
        ["echo", "inside_async"],
        capture_output=True,
        resources={"num_cpus": 1},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == b"inside_async"
    assert pul.is_initialized()
    assert _rt.owns_system() is True
    assert get_shared_loop() is not None
    _assert_has_bound_addr()


@pytest.mark.asyncio
async def test_async_context_with_explicit_init_supports_threaded_sync_resources(
    monkeypatch,
):
    _require_ray()
    _enable_pulsing_backend(monkeypatch)
    await pul.init(addr="127.0.0.1:0")

    result = await asyncio.to_thread(
        run,
        ["echo", "blocked"],
        capture_output=True,
        resources={"num_cpus": 1},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == b"blocked"
    assert get_shared_loop() is None
    assert _rt.owns_system() is False


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


def test_process_stdio_proxies_delegate_through_run_sync(monkeypatch):
    class FakeActorProxy:
        async def stdin_write(self, data):
            return ("stdin_write", data)

        async def stdin_flush(self):
            return "stdin_flush"

        async def stdin_close(self):
            return "stdin_close"

        async def stdin_fileno(self):
            return 10

        async def stdout_read(self, n=-1):
            return f"stdout_read:{n}"

        async def stdout_readline(self):
            return "stdout_readline"

        async def stdout_readlines(self):
            return ["stdout_readlines"]

        async def stdout_close(self):
            return "stdout_close"

        async def stdout_fileno(self):
            return 11

        async def stderr_read(self, n=-1):
            return f"stderr_read:{n}"

        async def stderr_readline(self):
            return "stderr_readline"

        async def stderr_readlines(self):
            return ["stderr_readlines"]

        async def stderr_close(self):
            return "stderr_close"

        async def stderr_fileno(self):
            return 12

    monkeypatch.setattr(
        process_module, "run_sync", lambda awaitable: asyncio.run(awaitable)
    )

    proxy = FakeActorProxy()
    stdin = process_module._StdinProxy(proxy)
    stdout = process_module._StdoutProxy(proxy)
    stderr = process_module._StderrProxy(proxy)

    stdin_write_result = stdin.write(b"hello")
    assert stdin_write_result is None
    stdin_flush_result = stdin.flush()
    assert stdin_flush_result is None
    stdin_close_result = stdin.close()
    assert stdin_close_result is None
    assert stdin.fileno() == 10

    assert stdout.read(3) == "stdout_read:3"
    assert stdout.readline() == "stdout_readline"
    assert stdout.readlines() == ["stdout_readlines"]
    stdout_close_result = stdout.close()
    assert stdout_close_result is None
    assert stdout.fileno() == 11

    assert stderr.read(5) == "stderr_read:5"
    assert stderr.readline() == "stderr_readline"
    assert stderr.readlines() == ["stderr_readlines"]
    stderr_close_result = stderr.close()
    assert stderr_close_result is None
    assert stderr.fileno() == 12


def test_popen_remote_mode_uses_run_sync_for_spawn_and_methods(monkeypatch):
    calls = []

    class FakeProxy:
        def __init__(self):
            self._ray_node_actor = None

        async def poll(self):
            calls.append(("poll",))
            return 3

        async def wait(self, timeout):
            calls.append(("wait", timeout))
            return 4

        async def communicate(self, input, timeout):
            calls.append(("communicate", input, timeout))
            return b"out", b"err"

        async def returncode(self):
            calls.append(("returncode",))
            return 5

        async def send_signal(self, sig):
            calls.append(("send_signal", sig))

        async def terminate(self):
            calls.append(("terminate",))

        async def kill(self):
            calls.append(("kill",))

    async def fake_aspawn(self):
        self._proxy = FakeProxy()
        self.pid = 1234
        self.stdin = SimpleNamespace()
        self.stdout = SimpleNamespace()
        self.stderr = SimpleNamespace()

    ensure_calls = []

    monkeypatch.setattr(popen_module, "_should_use_pulsing", lambda resources: True)
    monkeypatch.setattr(
        popen_module,
        "ensure_sync_runtime",
        lambda **kwargs: ensure_calls.append(kwargs),
    )
    monkeypatch.setattr(
        popen_module, "run_sync", lambda awaitable: asyncio.run(awaitable)
    )
    monkeypatch.setattr(popen_module.Popen, "_aspawn", fake_aspawn)

    proc = Popen(["fake"], resources={"num_cpus": 1})

    assert proc.pid == 1234
    assert ensure_calls == [{"addr": "0.0.0.0:0", "require_addr": True}]
    poll_result = proc.poll()
    assert poll_result == 3
    wait_result = proc.wait(timeout=1.5)
    assert wait_result == 4
    communicate_result = proc.communicate(input=b"in", timeout=2.0)
    assert communicate_result == (b"out", b"err")
    assert proc.returncode == 5

    proc.send_signal(9)
    proc.terminate()
    proc.kill()

    assert calls == [
        ("poll",),
        ("wait", 1.5),
        ("communicate", b"in", 2.0),
        ("returncode",),
        ("send_signal", 9),
        ("terminate",),
        ("kill",),
    ]


def test_popen_close_cleans_up_remote_actor(monkeypatch):
    cleanup_calls = []

    async def fake_cleanup(actor_proxy):
        cleanup_calls.append(actor_proxy)
        actor_proxy._ray_node_actor = None

    proxy = SimpleNamespace(_ray_node_actor=object())
    proc = Popen.__new__(Popen)
    proc._native = None
    proc._proxy = proxy

    monkeypatch.setattr(
        popen_module, "run_sync", lambda awaitable: asyncio.run(awaitable)
    )
    monkeypatch.setattr(
        "pulsing.subprocess.ray_spawn.cleanup_ray_actor",
        fake_cleanup,
    )

    proc.close()

    assert cleanup_calls == [proxy]
    assert proc._proxy._ray_node_actor is None


def test_popen_closes_spawn_coroutine_when_run_sync_fails(monkeypatch):
    captured = {}

    async def fake_aspawn(self):
        return None

    def fail_run_sync(awaitable):
        captured["awaitable"] = awaitable
        raise RuntimeError("spawn boom")

    monkeypatch.setattr(popen_module, "_should_use_pulsing", lambda resources: True)
    monkeypatch.setattr(popen_module, "ensure_sync_runtime", lambda **kwargs: None)
    monkeypatch.setattr(popen_module, "run_sync", fail_run_sync)
    monkeypatch.setattr(popen_module.Popen, "_aspawn", fake_aspawn)

    with pytest.raises(RuntimeError, match="spawn boom"):
        Popen(["fake"], resources={"num_cpus": 1})

    assert captured["awaitable"].cr_frame is None
