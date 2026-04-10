"""Tests for internal async bridge helpers."""

from __future__ import annotations

import asyncio
import threading

import pytest

from pulsing._async_bridge import (
    bind_run_sync,
    clear_pulsing_loop,
    get_loop,
    get_running_loop,
    get_sync_auto_init_loop,
    get_pulsing_loop,
    get_shared_loop,
    get_shared_loop_state,
    get_shared_thread,
    run_sync,
    run_in_worker,
    select_loop,
    set_pulsing_loop,
    start_shared_loop,
    stop_shared_loop,
    submit_on_shared_loop,
    submit_to_loop,
    valid_loop,
)


@pytest.fixture(autouse=True)
def reset_shared_loop():
    stop_shared_loop()
    clear_pulsing_loop()
    yield
    stop_shared_loop()
    clear_pulsing_loop()


def test_start_shared_loop_is_singleton():
    first = start_shared_loop()
    second = start_shared_loop()
    state_loop, state_thread = get_shared_loop_state()

    assert first is second
    assert get_shared_loop() is first
    assert get_shared_thread() is not None
    assert state_loop is first
    assert state_thread is get_shared_thread()


def test_get_sync_auto_init_loop_reuses_shared_loop():
    first = get_sync_auto_init_loop()
    second = get_sync_auto_init_loop()

    assert first is second
    assert get_shared_loop() is first


def test_submit_on_shared_loop_returns_result():
    async def add(a: int, b: int) -> int:
        return a + b

    loop = start_shared_loop()
    assert submit_on_shared_loop(add(1, 2), timeout=1) == 3
    assert submit_to_loop(loop, add(2, 3), timeout=1) == 5


def test_submit_on_shared_loop_supports_timeout():
    async def sleeper() -> None:
        await asyncio.sleep(0.2)

    start_shared_loop()
    with pytest.raises(TimeoutError):
        submit_on_shared_loop(sleeper(), timeout=0.01)


def test_submit_to_loop_supports_timeout():
    async def sleeper() -> None:
        await asyncio.sleep(0.2)

    loop = start_shared_loop()
    with pytest.raises(TimeoutError):
        submit_to_loop(loop, sleeper(), timeout=0.01)


def test_stop_shared_loop_is_idempotent_and_clears_state():
    start_shared_loop()

    stop_shared_loop()
    stop_shared_loop()

    assert get_shared_loop() is None
    assert get_shared_thread() is None


def test_pulsing_loop_set_get_and_clear():
    loop = asyncio.new_event_loop()
    try:
        set_pulsing_loop(loop)
        assert get_pulsing_loop() is None

        async def mark_loop():
            set_pulsing_loop(asyncio.get_running_loop())
            assert get_pulsing_loop() is asyncio.get_running_loop()

        loop.run_until_complete(mark_loop())
    finally:
        clear_pulsing_loop()
        loop.close()

    assert get_pulsing_loop() is None


def test_select_loop_returns_first_valid_loop():
    loop = start_shared_loop()

    assert select_loop(None, loop) is loop
    assert select_loop(loop, None) is loop
    assert select_loop(None, None) is None


def test_run_sync_submits_to_selected_loop():
    async def add(a: int, b: int) -> int:
        return a + b

    loop = start_shared_loop()
    assert run_sync(add(4, 5), loop=loop, timeout=1) == 9


def test_run_sync_accepts_loop_getter():
    async def add(a: int, b: int) -> int:
        return a + b

    start_shared_loop()
    assert run_sync(add(5, 6), loop=get_shared_loop, timeout=1) == 11


def test_run_sync_same_loop_raise():
    loop = asyncio.new_event_loop()

    async def inner() -> None:
        with pytest.raises(RuntimeError, match="same event loop"):
            run_sync(
                asyncio.sleep(0),
                loop=asyncio.get_running_loop(),
                same_loop="raise",
                same_loop_message="same event loop",
            )

    try:
        loop.run_until_complete(inner())
    finally:
        loop.close()


def test_get_sync_auto_init_loop_same_thread_raise():
    loop = asyncio.new_event_loop()

    async def inner() -> None:
        with pytest.raises(RuntimeError, match="same thread"):
            get_sync_auto_init_loop(same_thread_message="same thread")

    try:
        loop.run_until_complete(inner())
    finally:
        loop.close()


def test_run_sync_missing_loop_raise():
    with pytest.raises(RuntimeError, match="Event loop not running"):
        run_sync(
            asyncio.sleep(0),
            loop=None,
            missing_loop="raise",
            missing_loop_message="Event loop not running",
        )


def test_run_sync_uses_worker_inside_running_loop():
    async def inner() -> int:
        return run_sync(asyncio.sleep(0, result=11))

    assert asyncio.run(inner()) == 11


def test_bind_run_sync_uses_fixed_loop():
    async def add(a: int, b: int) -> int:
        return a + b

    loop = start_shared_loop()
    bound = bind_run_sync(loop=loop, timeout=1)
    assert bound(add(6, 7)) == 13


def test_bind_run_sync_supports_loop_getter():
    async def add(a: int, b: int) -> int:
        return a + b

    start_shared_loop()
    bound = bind_run_sync(loop=get_shared_loop, timeout=1)
    assert bound(add(7, 8)) == 15


def test_run_sync_reuses_explicit_pulsing_loop_from_another_thread():
    ready = threading.Event()
    stop = threading.Event()

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def keep_alive() -> None:
            set_pulsing_loop(asyncio.get_running_loop())
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)

        try:
            loop.run_until_complete(keep_alive())
        finally:
            clear_pulsing_loop()
            loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)

    try:
        assert (
            run_sync(asyncio.sleep(0, result=23), loop=get_pulsing_loop, timeout=1)
            == 23
        )
    finally:
        stop.set()
        thread.join(timeout=5)


def test_bind_run_sync_same_loop_raise():
    loop = asyncio.new_event_loop()

    async def inner() -> None:
        bound = bind_run_sync(
            loop=asyncio.get_running_loop,
            same_loop="raise",
            same_loop_message="same event loop",
        )
        with pytest.raises(RuntimeError, match="same event loop"):
            bound(asyncio.sleep(0))

    try:
        loop.run_until_complete(inner())
    finally:
        loop.close()


def test_bind_run_sync_uses_worker_inside_running_loop():
    bound = bind_run_sync()

    async def inner() -> int:
        return bound(asyncio.sleep(0, result=19))

    assert asyncio.run(inner()) == 19


def test_run_in_worker_inside_running_loop():
    async def inner() -> tuple[int, bool]:
        def call_sync() -> tuple[int, bool]:
            async def compute() -> tuple[int, bool]:
                return 7, get_running_loop() is not None

            return run_in_worker(compute(), timeout=1)

        return await asyncio.to_thread(call_sync)

    value, has_loop = asyncio.run(inner())
    assert value == 7
    assert has_loop is True


def test_get_running_loop_outside_async_returns_none():
    assert get_running_loop() is None


def test_valid_loop_rejects_stopped_loop():
    loop = asyncio.new_event_loop()
    try:
        assert valid_loop(loop) is None
    finally:
        loop.close()


def test_get_loop_returns_user_loop_when_set():
    """get_loop() prefers the user (pulsing) loop over the shared loop."""
    shared = start_shared_loop()

    assert get_loop() is shared

    ready = threading.Event()
    stop = threading.Event()

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def keep_alive() -> None:
            set_pulsing_loop(asyncio.get_running_loop())
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)

        try:
            loop.run_until_complete(keep_alive())
        finally:
            clear_pulsing_loop()
            loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)

    try:
        user_loop = get_pulsing_loop()
        assert user_loop is not None
        assert get_loop() is user_loop
    finally:
        stop.set()
        thread.join(timeout=5)


def test_get_loop_returns_none_when_nothing_running():
    assert get_loop() is None
