"""Internal helpers for background event-loop bridging."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable

LoopRef = (
    asyncio.AbstractEventLoop | None | Callable[[], asyncio.AbstractEventLoop | None]
)


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


def submit_to_loop(
    loop: asyncio.AbstractEventLoop,
    awaitable,
    *,
    timeout: float | None = None,
    ensure_coro: bool = False,
) -> Any:
    """Submit an awaitable to an existing loop and block for the result."""
    if ensure_coro:
        awaitable = _ensure_coro(awaitable)
    future = asyncio.run_coroutine_threadsafe(awaitable, loop)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError() from exc


def run_in_worker(
    awaitable,
    *,
    timeout: float | None = None,
    ensure_coro: bool = False,
) -> Any:
    """Run an awaitable in a fresh worker thread with its own event loop."""
    if ensure_coro:
        awaitable = _ensure_coro(awaitable)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, awaitable).result(timeout=timeout)


class BackgroundLoopRunner:
    """Manage a background event loop stored in caller-owned state."""

    def __init__(
        self,
        *,
        thread_name: str,
        get_loop: Callable[[], asyncio.AbstractEventLoop | None],
        set_loop: Callable[[asyncio.AbstractEventLoop | None], None],
        get_thread: Callable[[], threading.Thread | None],
        set_thread: Callable[[threading.Thread | None], None],
    ) -> None:
        self._thread_name = thread_name
        self._get_loop = get_loop
        self._set_loop = set_loop
        self._get_thread = get_thread
        self._set_thread = set_thread
        self._lock = threading.Lock()

    def start(self) -> asyncio.AbstractEventLoop:
        """Start the background loop if needed and return it."""
        loop = valid_loop(self._get_loop())
        if loop is not None:
            return loop

        with self._lock:
            loop = valid_loop(self._get_loop())
            if loop is not None:
                return loop

            ready = threading.Event()

            def _run() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._set_loop(loop)
                ready.set()
                loop.run_forever()

                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

                self._set_loop(None)
                self._set_thread(None)

            thread = threading.Thread(
                target=_run,
                daemon=True,
                name=self._thread_name,
            )
            self._set_thread(thread)
            thread.start()
            ready.wait()

            loop = self._get_loop()
            if loop is None:
                raise RuntimeError(
                    f"Failed to start background event loop {self._thread_name!r}."
                )
            return loop

    def submit(
        self,
        awaitable,
        *,
        timeout: float | None = None,
        ensure_coro: bool = False,
    ) -> Any:
        """Submit an awaitable to the managed background loop."""
        return submit_to_loop(
            self.start(),
            awaitable,
            timeout=timeout,
            ensure_coro=ensure_coro,
        )

    def stop(self, *, join_timeout: float = 5.0) -> None:
        """Request loop shutdown and wait for the worker thread to exit."""
        with self._lock:
            loop = self._get_loop()
            thread = self._get_thread()

            if loop is not None and loop.is_running():
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    pass

            if thread is not None:
                thread.join(timeout=join_timeout)

            if self._get_thread() is thread:
                self._set_thread(None)
            if self._get_loop() is loop:
                self._set_loop(None)


_shared_loop: asyncio.AbstractEventLoop | None = None
_shared_thread: threading.Thread | None = None
_user_loop: asyncio.AbstractEventLoop | None = None


def _get_shared_loop_storage() -> asyncio.AbstractEventLoop | None:
    return _shared_loop


def _set_shared_loop_storage(loop: asyncio.AbstractEventLoop | None) -> None:
    global _shared_loop
    _shared_loop = loop


def _get_shared_thread_storage() -> threading.Thread | None:
    return _shared_thread


def _set_shared_thread_storage(thread: threading.Thread | None) -> None:
    global _shared_thread
    _shared_thread = thread


_shared_runner = BackgroundLoopRunner(
    thread_name="pulsing-background-loop",
    get_loop=_get_shared_loop_storage,
    set_loop=_set_shared_loop_storage,
    get_thread=_get_shared_thread_storage,
    set_thread=_set_shared_thread_storage,
)


def start_shared_loop() -> asyncio.AbstractEventLoop:
    """Start the shared background event loop and return it."""
    return _shared_runner.start()


def submit_on_shared_loop(
    awaitable,
    *,
    timeout: float | None = None,
    ensure_coro: bool = False,
) -> Any:
    """Submit work to the shared background event loop."""
    return _shared_runner.submit(
        awaitable,
        timeout=timeout,
        ensure_coro=ensure_coro,
    )


def stop_shared_loop(*, join_timeout: float = 5.0) -> None:
    """Stop the shared background event loop if it is running."""
    _shared_runner.stop(join_timeout=join_timeout)


def get_shared_loop() -> asyncio.AbstractEventLoop | None:
    """Return the shared background loop when it is alive."""
    return valid_loop(_shared_loop)


def get_shared_thread() -> threading.Thread | None:
    """Return the shared background worker thread."""
    return _shared_thread


def get_shared_loop_state() -> tuple[
    asyncio.AbstractEventLoop | None,
    threading.Thread | None,
]:
    """Return the shared background loop and worker thread."""
    return get_shared_loop(), get_shared_thread()


def get_sync_auto_init_loop(
    *,
    same_thread_message: str | None = None,
) -> asyncio.AbstractEventLoop:
    """Return the shared loop for sync auto-init, or fail on same-thread async callers."""
    running_loop = get_running_loop()
    if running_loop is None:
        return start_shared_loop()

    raise RuntimeError(
        same_thread_message
        or "Sync auto-init cannot run on the same thread as a running event loop."
    )


def set_user_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Record the event loop that owns the global Pulsing system."""
    global _user_loop
    _user_loop = loop


def clear_user_loop() -> None:
    """Clear the recorded global Pulsing owning loop."""
    set_user_loop(None)


def _get_user_loop() -> asyncio.AbstractEventLoop | None:
    """Return the global Pulsing owning loop when it is alive."""
    return valid_loop(_user_loop)


def get_loop() -> asyncio.AbstractEventLoop | None:
    """Return the best available event loop: user loop first, shared loop second."""
    return valid_loop(_user_loop) or valid_loop(_shared_loop)


# Deprecated aliases — kept for backwards compatibility with tests and external code.
set_pulsing_loop = set_user_loop
clear_pulsing_loop = clear_user_loop
get_pulsing_loop = _get_user_loop


def select_loop(
    *loops: asyncio.AbstractEventLoop | None,
) -> asyncio.AbstractEventLoop | None:
    """Return the first alive event loop from *loops*."""
    for loop in loops:
        loop = valid_loop(loop)
        if loop is not None:
            return loop
    return None


def _close_awaitable(awaitable) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


def _resolve_loop(loop: LoopRef) -> asyncio.AbstractEventLoop | None:
    if callable(loop):
        loop = loop()
    return valid_loop(loop)


def run_sync(
    awaitable,
    *,
    loop: LoopRef = None,
    timeout: float | None = None,
    ensure_coro: bool = False,
    same_loop: str = "worker",
    same_loop_message: str | None = None,
    missing_loop: str = "run",
    missing_loop_message: str | None = None,
) -> Any:
    """Synchronously execute *awaitable* against the chosen event loop policy."""
    dispatch_loop = _resolve_loop(loop)
    running_loop = get_running_loop()

    if dispatch_loop is not None:
        if running_loop is dispatch_loop:
            if same_loop == "worker":
                return run_in_worker(
                    awaitable, timeout=timeout, ensure_coro=ensure_coro
                )
            if same_loop == "raise":
                _close_awaitable(awaitable)
                raise RuntimeError(
                    same_loop_message
                    or "Cannot block on the same event loop that owns this awaitable."
                )
            raise ValueError(f"Unsupported same_loop policy: {same_loop!r}")

        return submit_to_loop(
            dispatch_loop,
            awaitable,
            timeout=timeout,
            ensure_coro=ensure_coro,
        )

    if missing_loop == "raise":
        _close_awaitable(awaitable)
        raise RuntimeError(
            missing_loop_message or "Event loop not running for synchronous bridge."
        )

    if missing_loop != "run":
        raise ValueError(f"Unsupported missing_loop policy: {missing_loop!r}")

    if running_loop is not None:
        return run_in_worker(awaitable, timeout=timeout, ensure_coro=ensure_coro)

    if ensure_coro:
        awaitable = _ensure_coro(awaitable)
    return asyncio.run(awaitable)


def bind_run_sync(
    *,
    loop: LoopRef = None,
    timeout: float | None = None,
    ensure_coro: bool = False,
    same_loop: str = "worker",
    same_loop_message: str | None = None,
    missing_loop: str = "run",
    missing_loop_message: str | None = None,
) -> Callable[[Any], Any]:
    """Bind a reusable sync bridge callable with fixed execution policy."""

    def _bound(awaitable) -> Any:
        return run_sync(
            awaitable,
            loop=loop,
            timeout=timeout,
            ensure_coro=ensure_coro,
            same_loop=same_loop,
            same_loop_message=same_loop_message,
            missing_loop=missing_loop,
            missing_loop_message=missing_loop_message,
        )

    return _bound
