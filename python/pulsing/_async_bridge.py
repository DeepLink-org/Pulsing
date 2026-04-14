"""Internal helpers for background event-loop bridging."""

from __future__ import annotations

import asyncio
import inspect
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any


async def _ensure_coro(obj) -> Any:
    """Await arbitrary awaitable objects via a real coroutine wrapper."""
    return await obj


def get_running_loop() -> asyncio.AbstractEventLoop | None:
    """Return the current running loop, or None outside async code."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def valid_loop(
    loop: asyncio.AbstractEventLoop | None,
) -> asyncio.AbstractEventLoop | None:
    """Return loop when it is alive and running, otherwise None."""
    if loop is None:
        return None
    if loop.is_closed() or not loop.is_running():
        return None
    return loop


_shared_loop: asyncio.AbstractEventLoop | None = None
_shared_thread: threading.Thread | None = None
_pulsing_loop: asyncio.AbstractEventLoop | None = None
_shared_lock = threading.Lock()


def _close_awaitable(awaitable) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


def _start_shared_loop() -> asyncio.AbstractEventLoop:
    global _shared_loop, _shared_thread

    ready = threading.Event()
    started: dict[str, asyncio.AbstractEventLoop | None] = {"loop": None}

    def _run() -> None:
        global _shared_loop, _shared_thread

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        started["loop"] = loop
        ready.set()

        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

            with _shared_lock:
                if _shared_loop is loop:
                    _shared_loop = None
                if _shared_thread is threading.current_thread():
                    _shared_thread = None

    with _shared_lock:
        loop = valid_loop(_shared_loop) or valid_loop(_pulsing_loop)
        if loop is not None:
            return loop

        thread = threading.Thread(
            target=_run,
            daemon=True,
            name="pulsing-background-loop",
        )
        thread.start()
        if not ready.wait(timeout=5):
            raise RuntimeError(
                "Failed to start background event loop 'pulsing-background-loop'."
            )

        loop = valid_loop(started["loop"])
        if loop is None:
            raise RuntimeError(
                "Failed to start background event loop 'pulsing-background-loop'."
            )

        _shared_loop = loop
        _shared_thread = thread
        return loop


def get_shared_loop() -> asyncio.AbstractEventLoop | None:
    """Return the owned shared background loop when it is alive."""
    return valid_loop(_shared_loop)


def get_pulsing_loop() -> asyncio.AbstractEventLoop | None:
    """Return the Pulsing owning loop when it is alive."""
    return valid_loop(_pulsing_loop)


def get_loop() -> asyncio.AbstractEventLoop | None:
    """Return the active Pulsing dispatch loop."""
    return get_shared_loop() or get_pulsing_loop()


def stop_shared_loop(*, join_timeout: float = 5.0) -> None:
    """Stop the owned shared background event loop if it is running."""
    global _shared_loop, _shared_thread

    with _shared_lock:
        loop = _shared_loop
        thread = _shared_thread

    if loop is None or thread is None:
        return

    if loop.is_running():
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

    if thread is threading.current_thread():
        return

    thread.join(timeout=join_timeout)

    with _shared_lock:
        if _shared_loop is loop:
            _shared_loop = None
        if _shared_thread is thread:
            _shared_thread = None


def set_pulsing_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Record the event loop that owns the global Pulsing system."""
    global _pulsing_loop

    with _shared_lock:
        shared_loop = valid_loop(_shared_loop)
        loop = valid_loop(loop)

        if _pulsing_loop is not None and loop is not _pulsing_loop:
            raise RuntimeError(
                "Pulsing loop is already set to a different active event loop."
            )
        _pulsing_loop = loop
        if (
            shared_loop is not None
            and _pulsing_loop is not None
            and _pulsing_loop is not shared_loop
        ):
            raise RuntimeError("Pulsing loop must match the active shared loop.")


def clear_pulsing_loop() -> None:
    """Clear the recorded global Pulsing owning loop."""
    global _pulsing_loop
    _pulsing_loop = None


def run_sync(
    awaitable,
    timeout: float | None = None,
) -> Any:
    """Synchronously execute *awaitable* using the active Pulsing dispatch loop."""
    if not inspect.iscoroutine(awaitable):
        awaitable = _ensure_coro(awaitable)

    submitted = False
    try:
        dispatch_loop = get_loop()
        if dispatch_loop is None:
            dispatch_loop = _start_shared_loop()

        if get_running_loop() is dispatch_loop:
            raise RuntimeError(
                "run_sync() cannot be called from the active Pulsing dispatch loop. "
                "Await the coroutine directly or move the sync API call to another "
                "thread."
            )

        future = asyncio.run_coroutine_threadsafe(awaitable, dispatch_loop)
        submitted = True
    except Exception:
        if not submitted:
            _close_awaitable(awaitable)
        raise

    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError() from exc
