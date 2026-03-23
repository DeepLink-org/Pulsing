"""Tests for Ray actor lifecycle cleanup in pulsing.subprocess."""

import asyncio

import pytest

ray = pytest.importorskip("ray")

import pulsing as pul
from pulsing.subprocess import PIPE, Popen, run
from pulsing.subprocess.ray_spawn import cleanup_ray_actor


@pytest.fixture(autouse=True)
async def ray_and_pulsing():
    """Start a local Ray cluster and Pulsing system, tear down in reverse order."""
    if not ray.is_initialized():
        ray.init(num_cpus=2)
    await pul.init(addr="127.0.0.1:0")
    yield
    await pul.shutdown()
    ray.shutdown()


def _in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, fn, *args)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_popen_close_cleans_up_ray_actor():
    """close() sets _ray_node_actor=None and the Ray actor is dead after."""

    def _test():
        proc = Popen(
            ["echo", "hello"],
            stdout=PIPE,
            resources={"num_cpus": 1},
        )
        proc.communicate()
        handle = proc._proxy._ray_node_actor
        assert handle is not None

        proc.close()
        assert proc._proxy._ray_node_actor is None

        # The Ray actor should be dead — calling a method should raise
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(handle.shutdown.remote())

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_popen_context_manager_cleans_up_ray():
    """Using `with Popen(...)` triggers cleanup on exit."""

    def _test():
        with Popen(
            ["echo", "ctx"],
            stdout=PIPE,
            resources={"num_cpus": 1},
        ) as proc:
            proc.communicate()
            handle = proc._proxy._ray_node_actor
            assert handle is not None

        # After context manager exit, handle should be cleared
        assert proc._proxy._ray_node_actor is None

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_double_close_is_noop():
    """Second close() does not raise."""

    def _test():
        proc = Popen(
            ["echo", "twice"],
            stdout=PIPE,
            resources={"num_cpus": 1},
        )
        proc.communicate()
        proc.close()
        proc.close()  # should not raise

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_close_noop_for_local_process():
    """close() is harmless for non-Ray (local) processes."""

    def _test():
        proc = Popen(["echo", "local"], stdout=PIPE)
        proc.communicate()
        proc.close()  # should not raise
        assert proc.returncode == 0

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_cleanup_ray_actor_directly():
    """cleanup_ray_actor(proxy) works on a proxy from ray_spawn_process_actor()."""

    def _test():
        proc = Popen(
            ["echo", "direct"],
            stdout=PIPE,
            resources={"num_cpus": 1},
        )
        proc.communicate()

        from pulsing.subprocess.popen import _run_sync

        handle = proc._proxy._ray_node_actor
        assert handle is not None
        _run_sync(cleanup_ray_actor(proc._proxy))
        assert proc._proxy._ray_node_actor is None

    await _in_executor(_test)


@pytest.mark.asyncio
async def test_run_helper_cleans_up_ray():
    """High-level run() cleans up the Ray actor after completion."""

    def _test():
        # run() should not leak — we can't easily check the handle
        # after run() returns, but we verify it completes without error.
        result = run(
            ["echo", "run_cleanup"],
            capture_output=True,
            resources={"num_cpus": 1},
        )
        assert result.returncode == 0
        assert b"run_cleanup" in result.stdout

    await _in_executor(_test)
