"""Bridge Pulsing runtime telemetry to probing via memtable (mmap'd ring buffers).

All data is written by Rust directly to mmap files under ``<tmpdir>/probing/<pid>/``,
discovered automatically by probing's DataFusion engine:

- ``pulsing.actors``  — live actor registry (MEMH hash table)
- ``pulsing.spans``   — completed OTel spans (MEMT ring buffer)
- ``pulsing.metrics`` — periodic system metric snapshots (MEMT ring buffer)
- ``pulsing.members`` — cluster membership state (MEMH hash table)

The Python background thread triggers periodic ``get_metrics()`` calls so
the Rust side records fresh metric snapshots. Member events are written
automatically by the gossip protocol on every state transition.

Environment:

- ``PULSING_PROBING_INTEGRATION``: set to ``0`` / ``false`` / ``no`` to disable.
- ``PULSING_PROBING_REFRESH_SEC``: refresh interval for the background thread (default 10).
- ``PULSING_PROBING_AUTO_TRACING``: default ``1`` — if ``PULSING_TRACING`` is not set, still
  install a **silent** Rust OTel subscriber so span / actor memtables get populated.
  Set to ``0`` to disable.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_refresh_thread: threading.Thread | None = None
_stop = threading.Event()


def _integration_disabled() -> bool:
    v = os.environ.get("PULSING_PROBING_INTEGRATION", "1").strip().lower()
    return v in ("0", "false", "no")


def _refresh_interval_sec() -> float:
    try:
        return float(os.environ.get("PULSING_PROBING_REFRESH_SEC", "10"))
    except ValueError:
        return 10.0


def ensure_tracing_for_span_capture() -> None:
    """Install Rust OTel subscriber so memtable span/actor data gets populated.

    No-op when ``PULSING_TRACING=1`` (tracing already active) or
    ``PULSING_PROBING_AUTO_TRACING=0``.
    """
    v = os.environ.get("PULSING_PROBING_AUTO_TRACING", "1").strip().lower()
    if v in ("0", "false", "no"):
        return
    if os.environ.get("PULSING_TRACING", "").strip().lower() in ("1", "true", "yes"):
        return
    try:
        from pulsing._core import init_distributed_tracing

        init_distributed_tracing(
            service_name=os.environ.get("PULSING_SERVICE_NAME") or None,
            console_output=False,
        )
    except Exception as e:
        logger.debug("ensure_tracing_for_span_capture failed: %s", e)


async def _trigger_snapshots() -> None:
    """Call Rust APIs that record metric snapshots to memtable."""
    from pulsing.core import get_system, get_system_actor, is_initialized

    if not is_initialized():
        return

    ensure_tracing_for_span_capture()

    system = get_system()
    proxy = await get_system_actor(system)
    await proxy.get_metrics()


def _background_loop() -> None:
    import asyncio

    interval = _refresh_interval_sec()
    while True:
        try:
            asyncio.run(_trigger_snapshots())
        except Exception as e:
            logger.debug("Pulsing probing integration refresh failed: %s", e)
        if _stop.wait(timeout=interval):
            break


def start_probing_integration() -> bool:
    """Start a daemon thread that periodically triggers memtable snapshot writes."""
    global _refresh_thread

    if _integration_disabled():
        return False
    if _refresh_thread is not None and _refresh_thread.is_alive():
        return True

    ensure_tracing_for_span_capture()

    _stop.clear()
    _refresh_thread = threading.Thread(
        target=_background_loop,
        name="pulsing-probing-integration",
        daemon=True,
    )
    _refresh_thread.start()
    return True


def stop_probing_integration() -> None:
    """Stop the background refresh thread."""
    global _refresh_thread

    _stop.set()
    t = _refresh_thread
    if t is not None and t.is_alive():
        t.join(timeout=3.0)
    _refresh_thread = None


async def refresh_probing_tables_async() -> bool:
    """Trigger one metric snapshot refresh cycle (metrics → memtable)."""
    if _integration_disabled():
        return False
    try:
        await _trigger_snapshots()
    except Exception as e:
        logger.debug("Pulsing probing async refresh failed: %s", e)
        return False
    return True
