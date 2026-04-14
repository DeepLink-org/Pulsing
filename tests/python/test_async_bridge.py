"""Tests for internal async bridge helpers."""

from __future__ import annotations

import asyncio
import threading

import pytest

import pulsing._async_bridge as bridge
from pulsing._async_bridge import (
    clear_pulsing_loop,
    get_loop,
    get_pulsing_loop,
    get_running_loop,
    get_shared_loop,
    run_sync,
    set_pulsing_loop,
    stop_shared_loop,
    valid_loop,
)


@pytest.fixture(autouse=True)
def reset_bridge_state():
    stop_shared_loop()
    clear_pulsing_loop()
    yield
    stop_shared_loop()
    clear_pulsing_loop()


def _start_loop_thread(
    *,
    set_as_pulsing: bool = False,
) -> tuple[asyncio.AbstractEventLoop, threading.Event, threading.Thread]:
    ready = threading.Event()
    stop = threading.Event()
    holder: dict[str, asyncio.AbstractEventLoop] = {}

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def keep_alive() -> None:
            running_loop = asyncio.get_running_loop()
            holder["loop"] = running_loop
            if set_as_pulsing:
                set_pulsing_loop(running_loop)
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)

        try:
            loop.run_until_complete(keep_alive())
        finally:
            if set_as_pulsing:
                clear_pulsing_loop()
            loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    return holder["loop"], stop, thread


def test_run_sync_creates_owned_loop_and_reuses_it():
    async def current_loop():
        return asyncio.get_running_loop()

    first = run_sync(current_loop(), timeout=1)
    second = run_sync(current_loop(), timeout=1)

    assert first is second
    assert get_shared_loop() is first
    assert get_pulsing_loop() is None
    assert get_loop() is first


def test_run_sync_uses_pulsing_loop_when_shared_loop_is_absent():
    pulsing_loop, stop, thread = _start_loop_thread(set_as_pulsing=True)

    async def current_loop():
        return asyncio.get_running_loop()

    try:
        assert run_sync(current_loop(), timeout=1) is pulsing_loop
        assert get_shared_loop() is None
        assert get_pulsing_loop() is pulsing_loop
    finally:
        stop.set()
        thread.join(timeout=5)


def test_set_pulsing_loop_rejects_different_running_shared_loop():
    run_sync(asyncio.sleep(0), timeout=1)
    other_loop, stop, thread = _start_loop_thread()

    try:
        with pytest.raises(RuntimeError, match="must match the active shared loop"):
            set_pulsing_loop(other_loop)
    finally:
        stop.set()
        thread.join(timeout=5)


def test_stop_shared_loop_clears_owned_loop_state():
    run_sync(asyncio.sleep(0), timeout=1)

    stop_shared_loop()

    assert get_shared_loop() is None
    assert get_loop() is None


def test_stop_shared_loop_is_noop_for_init_first_mode():
    pulsing_loop, stop, thread = _start_loop_thread(set_as_pulsing=True)

    try:
        stop_shared_loop()
        assert get_shared_loop() is None
        assert get_pulsing_loop() is pulsing_loop
        assert get_loop() is pulsing_loop
    finally:
        stop.set()
        thread.join(timeout=5)


def test_run_sync_submits_to_dispatch_loop():
    async def current_loop():
        return asyncio.get_running_loop()

    loop = run_sync(current_loop(), timeout=1)
    assert loop is get_shared_loop()
    assert run_sync(asyncio.sleep(0, result=9), timeout=1) == 9


def test_run_sync_uses_pulsing_loop_from_another_thread():
    _, stop, thread = _start_loop_thread(set_as_pulsing=True)

    try:
        assert run_sync(asyncio.sleep(0, result=23), timeout=1) == 23
    finally:
        stop.set()
        thread.join(timeout=5)


def test_run_sync_raises_on_active_dispatch_loop():
    loop, stop, thread = _start_loop_thread(set_as_pulsing=True)

    async def call_run_sync() -> None:
        with pytest.raises(RuntimeError, match="active Pulsing dispatch loop"):
            run_sync(asyncio.sleep(0, result=11), timeout=1)

    try:
        asyncio.run_coroutine_threadsafe(call_run_sync(), loop).result(timeout=1)
    finally:
        stop.set()
        thread.join(timeout=5)


def test_run_sync_creates_shared_loop_without_dispatch_loop():
    async def current_loop():
        return asyncio.get_running_loop()

    assert get_shared_loop() is None
    assert run_sync(asyncio.sleep(0, result=13), timeout=1) == 13

    shared_loop = get_shared_loop()
    assert shared_loop is not None
    assert get_loop() is shared_loop
    assert run_sync(current_loop(), timeout=1) is shared_loop


def test_run_sync_wraps_non_coroutine_awaitable():
    class AwaitableValue:
        def __await__(self):
            return asyncio.sleep(0, result=17).__await__()

    assert run_sync(AwaitableValue(), timeout=1) == 17


def test_run_sync_timeout_cancels_submitted_future():
    cancelled = threading.Event()

    async def sleeper() -> None:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(TimeoutError):
        run_sync(sleeper(), timeout=0.01)

    assert cancelled.wait(timeout=1)


def test_get_running_loop_outside_async_returns_none():
    assert get_running_loop() is None


def test_valid_loop_rejects_stopped_loop():
    loop = asyncio.new_event_loop()
    try:
        assert valid_loop(loop) is None
    finally:
        loop.close()


def test_get_loop_returns_none_when_nothing_running():
    assert get_loop() is None


def test_start_shared_loop_returns_existing_shared_loop():
    first = bridge._start_shared_loop()
    second = bridge._start_shared_loop()

    assert second is first


def test_start_shared_loop_cancels_pending_tasks_on_shutdown():
    loop = bridge._start_shared_loop()
    started = threading.Event()
    cancelled = threading.Event()

    async def sleeper() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    future = asyncio.run_coroutine_threadsafe(sleeper(), loop)
    assert started.wait(timeout=1)

    stop_shared_loop()

    assert cancelled.wait(timeout=1)
    assert future.cancelled()


def test_start_shared_loop_raises_when_ready_never_signals(monkeypatch):
    monkeypatch.setattr(bridge.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(
        bridge.threading.Event, "wait", lambda self, timeout=None: False
    )

    with pytest.raises(RuntimeError, match="Failed to start background event loop"):
        bridge._start_shared_loop()


def test_start_shared_loop_raises_when_started_loop_is_invalid(monkeypatch):
    monkeypatch.setattr(bridge.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(bridge.threading.Event, "wait", lambda self, timeout=None: True)

    with pytest.raises(RuntimeError, match="Failed to start background event loop"):
        bridge._start_shared_loop()


def test_stop_shared_loop_swallows_stop_errors_and_clears_state(monkeypatch):
    class FakeLoop:
        def is_running(self) -> bool:
            return True

        def call_soon_threadsafe(self, callback) -> None:
            raise RuntimeError("boom")

    class FakeThread:
        def __init__(self) -> None:
            self.join_calls = 0

        def join(self, timeout=None) -> None:
            self.join_calls += 1

    fake_loop = FakeLoop()
    fake_thread = FakeThread()
    monkeypatch.setattr(bridge, "_shared_loop", fake_loop)
    monkeypatch.setattr(bridge, "_shared_thread", fake_thread)

    stop_shared_loop()

    assert fake_thread.join_calls == 1
    assert bridge._shared_loop is None
    assert bridge._shared_thread is None


def test_stop_shared_loop_returns_early_on_current_thread(monkeypatch):
    class FakeLoop:
        def is_running(self) -> bool:
            return False

    fake_loop = FakeLoop()
    current = threading.current_thread()
    monkeypatch.setattr(bridge, "_shared_loop", fake_loop)
    monkeypatch.setattr(bridge, "_shared_thread", current)

    stop_shared_loop()

    assert bridge._shared_loop is fake_loop
    assert bridge._shared_thread is current


def test_set_pulsing_loop_rejects_second_active_loop():
    first_loop, first_stop, first_thread = _start_loop_thread()
    second_loop, second_stop, second_thread = _start_loop_thread()

    try:
        set_pulsing_loop(first_loop)
        with pytest.raises(
            RuntimeError,
            match="already set to a different active event loop",
        ):
            set_pulsing_loop(second_loop)
    finally:
        clear_pulsing_loop()
        first_stop.set()
        second_stop.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)
