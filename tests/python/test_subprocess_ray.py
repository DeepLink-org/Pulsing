"""Tests for Ray-backed pulsing.subprocess cleanup and auto-init."""

import pytest

ray = pytest.importorskip("ray")

import pulsing as pul
from pulsing._async_bridge import (
    clear_pulsing_loop,
    get_shared_loop,
    stop_shared_loop,
    submit_on_shared_loop,
)
import pulsing._runtime as _rt
import pulsing.subprocess.popen as popen_module
from pulsing.subprocess import PIPE, Popen, run
from pulsing.subprocess.ray_spawn import cleanup_ray_actor


def _assert_has_bound_addr() -> None:
    addr = str(pul.get_system().addr)
    host, sep, port = addr.rpartition(":")
    assert sep == ":"
    assert host
    assert port.isdigit()


def _enable_pulsing_backend(monkeypatch, value: str = "1") -> None:
    monkeypatch.setenv("USE_POLSING_SUBPROCESS", value)


@pytest.fixture(autouse=True)
def clear_subprocess_env(monkeypatch):
    monkeypatch.delenv("USE_POLSING_SUBPROCESS", raising=False)


@pytest.fixture(autouse=True)
async def ray_and_subprocess_backend():
    if not ray.is_initialized():
        ray.init(num_cpus=2)
    stop_shared_loop()
    clear_pulsing_loop()
    yield
    _rt.shutdown()
    if pul.is_initialized():
        await pul.shutdown()
    stop_shared_loop()
    clear_pulsing_loop()
    ray.shutdown()


def test_non_empty_resources_with_env_disabled_stays_native():
    proc = Popen(
        ["echo", "native"],
        stdout=PIPE,
        resources={"num_cpus": 1},
    )
    stdout, _ = proc.communicate()

    assert stdout.strip() == b"native"
    assert proc._native is not None
    assert proc._proxy is None
    assert not pul.is_initialized()


def test_resources_backend_prefers_bootstrap_when_ray_is_ready(monkeypatch):
    _enable_pulsing_backend(monkeypatch)

    result = run(
        ["echo", "bootstrap"],
        capture_output=True,
        resources={"num_cpus": 1},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == b"bootstrap"
    assert pul.is_initialized()
    assert _rt.owns_system() is False
    assert get_shared_loop() is not None
    _assert_has_bound_addr()


def test_popen_close_cleans_up_ray_actor(monkeypatch):
    _enable_pulsing_backend(monkeypatch)
    proc = Popen(
        ["echo", "hello"],
        stdout=PIPE,
        resources={"num_cpus": 1},
    )
    proc.communicate()
    handle = proc._proxy._ray_node_actor

    assert handle is not None
    assert pul.is_initialized()
    _assert_has_bound_addr()

    proc.close()
    assert proc._proxy._ray_node_actor is None

    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(handle.shutdown.remote())


def test_popen_context_manager_cleans_up_ray(monkeypatch):
    _enable_pulsing_backend(monkeypatch)
    with Popen(
        ["echo", "ctx"],
        stdout=PIPE,
        resources={"num_cpus": 1},
    ) as proc:
        proc.communicate()
        handle = proc._proxy._ray_node_actor
        assert handle is not None

    assert proc._proxy._ray_node_actor is None


def test_double_close_is_noop(monkeypatch):
    _enable_pulsing_backend(monkeypatch)
    proc = Popen(
        ["echo", "twice"],
        stdout=PIPE,
        resources={"num_cpus": 1},
    )
    proc.communicate()
    proc.close()
    proc.close()


def test_cleanup_ray_actor_directly(monkeypatch):
    _enable_pulsing_backend(monkeypatch)
    proc = Popen(
        ["echo", "direct"],
        stdout=PIPE,
        resources={"num_cpus": 1},
    )
    proc.communicate()
    handle = proc._proxy._ray_node_actor

    assert handle is not None

    submit_on_shared_loop(cleanup_ray_actor(proc._proxy), timeout=30)
    assert proc._proxy._ray_node_actor is None


def test_run_helper_cleans_up_ray(monkeypatch):
    _enable_pulsing_backend(monkeypatch)
    result = run(
        ["echo", "run_cleanup"],
        capture_output=True,
        resources={"num_cpus": 1},
    )

    assert result.returncode == 0
    assert b"run_cleanup" in result.stdout
