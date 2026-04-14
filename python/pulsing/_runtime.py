"""Centralized auto-init, atexit, and module-owned runtime management.

Extracts the shared lazy-init / shutdown / atexit pattern previously
duplicated in ``subprocess/popen.py`` and ``transfer_queue/runtime.py``.
"""

from __future__ import annotations

import asyncio
import atexit
import threading

from pulsing._async_bridge import (
    get_loop,
    get_running_loop as _get_running_loop,
    get_shared_loop,
    run_sync as bridge_run_sync,
    set_pulsing_loop,
    stop_shared_loop,
)

_lock = threading.Lock()
_module_owns_system: bool = False
_atexit_registered: bool = False

_async_init_lock: asyncio.Lock | None = None
_async_init_lock_loop: asyncio.AbstractEventLoop | None = None


# ---------------------------------------------------------------------------
# Public accessors (test-friendly)
# ---------------------------------------------------------------------------


def owns_system() -> bool:
    """Return whether the module-level runtime owns the global Pulsing system."""
    return _module_owns_system


def is_cleanup_registered() -> bool:
    """Return whether the atexit cleanup handler has been registered."""
    return _atexit_registered


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _claim_module_runtime() -> None:
    """Mark the current global Pulsing system for module-managed cleanup."""
    global _module_owns_system
    _register_cleanup()
    _module_owns_system = True


def _register_cleanup() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(_shutdown_all)
    _atexit_registered = True


def _get_async_init_lock() -> asyncio.Lock:
    global _async_init_lock, _async_init_lock_loop
    loop = asyncio.get_running_loop()
    if _async_init_lock is None or _async_init_lock_loop is not loop:
        _async_init_lock = asyncio.Lock()
        _async_init_lock_loop = loop
    return _async_init_lock


# ---------------------------------------------------------------------------
# Sync / async lazy-init
# ---------------------------------------------------------------------------


def ensure_sync_runtime(
    *,
    addr: str | None = None,
    require_addr: bool = False,
) -> None:
    """Ensure a dispatch loop and global Pulsing system for sync callers.

    Parameters
    ----------
    addr:
        Address to pass to ``pul.init()`` when auto-initializing.
    require_addr:
        If ``True``, raise when an already-initialized system has no routable
        address (used by subprocess).
    """
    global _module_owns_system

    import pulsing as pul

    with _lock:
        if pul.is_initialized():
            if require_addr:
                system = pul.get_system()
                if system.addr is None:
                    raise RuntimeError(
                        "resources requires a routable Pulsing address. "
                        "Initialize Pulsing explicitly with pul.init(addr='0.0.0.0:0')."
                    )
            dispatch_loop = get_loop()
            if dispatch_loop is None:
                raise RuntimeError(
                    "Found an initialized Pulsing system, but its dispatch loop is "
                    "unavailable. Re-initialize Pulsing on a running loop or let the "
                    "sync API auto-initialize it again."
                )
            return

        if pul.bootstrap(wait_timeout=0):
            if require_addr:
                system = pul.get_system()
                if system.addr is None:
                    raise RuntimeError(
                        "resources requires a routable Pulsing address. "
                        "Initialize Pulsing explicitly with pul.init(addr='0.0.0.0:0')."
                    )
            if get_loop() is None:
                raise RuntimeError(
                    "Bootstrapped Pulsing, but no running dispatch loop was exposed."
                )
            _claim_module_runtime()
            return

        _claim_module_runtime()
        init_coro = pul.init(addr=addr) if addr else pul.init()
        try:
            bridge_run_sync(init_coro)
        except Exception:
            _module_owns_system = False
            raise


async def ensure_async_runtime() -> None:
    """Ensure a global Pulsing system for async callers."""
    global _module_owns_system

    import pulsing as pul

    if pul.is_initialized():
        return

    async with _get_async_init_lock():
        if pul.is_initialized():
            return

        set_pulsing_loop(_get_running_loop())
        if await asyncio.to_thread(pul.bootstrap, wait_timeout=0):
            _claim_module_runtime()
            return

        await pul.init()
        _claim_module_runtime()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def _shutdown_all(*, best_effort: bool = True) -> None:
    """Unified atexit / explicit shutdown handler."""
    global _module_owns_system

    import pulsing as pul

    with _lock:
        owns = _module_owns_system

        if owns and pul.is_initialized():
            try:
                bridge_run_sync(pul.shutdown())
            except Exception:
                if not best_effort:
                    raise

        if owns and get_shared_loop() is not None:
            try:
                stop_shared_loop(join_timeout=5)
            except Exception:
                pass

        _module_owns_system = False


def shutdown(*, best_effort: bool = True) -> None:
    """Public shutdown function.

    ``best_effort=True`` (default) swallows errors — suitable for atexit.
    ``best_effort=False`` re-raises — used by ``transfer_queue.shutdown()``.
    """
    _shutdown_all(best_effort=best_effort)


def clear_module_ownership() -> None:
    """Clear the module-owns-system flag.

    Called by ``core/__init__.py`` shutdown to reset state when the user
    explicitly shuts down the global system.
    """
    global _module_owns_system
    _module_owns_system = False
