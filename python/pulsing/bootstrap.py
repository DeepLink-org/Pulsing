"""
pulsing.bootstrap - Auto cluster formation; single API.

Background thread tries to form/join the Pulsing cluster using the chosen backend(s).
Call bootstrap(ray=..., torchrun=..., on_ready=callback, wait_timeout=...) — one standard API.

若不传 ray / torchrun，默认两者都为 True，即先试 Ray 再试 torchrun，相当于自动检测当前环境。

Usage:
    from pulsing import bootstrap

    bootstrap()                                    # 不传则自动检测：两个都试（默认）
    bootstrap(ray=True, torchrun=False)            # only try Ray
    bootstrap(ray=False, torchrun=True)            # only try torchrun
    bootstrap(on_ready=lambda system: ...)        # start + callback when ready
    if bootstrap(wait_timeout=30): ...             # start + block until ready
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

_bootstrap_thread: threading.Thread | None = None
_stop = threading.Event()
_on_ready_callbacks: list[Callable[..., None]] = []
_lock = threading.Lock()


def _try_init_once(*, ray: bool, torchrun: bool) -> bool:
    """Try to initialize Pulsing with the given backends. Return True if now initialized."""
    from pulsing.core import is_initialized

    if is_initialized():
        return True

    if ray:
        try:
            from pulsing.integrations.ray import init_in_ray

            init_in_ray()
            if is_initialized():
                logger.debug("Pulsing bootstrap: initialized via init_in_ray")
                return True
        except Exception as e:
            logger.debug("Pulsing bootstrap: init_in_ray skipped: %s", e)

    if torchrun:
        try:
            from pulsing.integrations.torchrun import init_in_torchrun

            init_in_torchrun()
            if is_initialized():
                logger.debug("Pulsing bootstrap: initialized via init_in_torchrun")
                return True
        except Exception as e:
            logger.debug("Pulsing bootstrap: init_in_torchrun skipped: %s", e)

    return False


def _fire_on_ready() -> None:
    from pulsing.core import get_system, is_initialized

    with _lock:
        callbacks = _on_ready_callbacks.copy()
        _on_ready_callbacks.clear()
    if not is_initialized():
        return
    system = get_system()
    for cb in callbacks:
        try:
            try:
                cb(system)
            except TypeError:
                cb()
        except Exception as e:
            logger.warning("bootstrap on_ready callback failed: %s", e)


def _bootstrap_loop(
    interval_sec: float,
    *,
    ray: bool,
    torchrun: bool,
) -> None:
    while not _stop.wait(timeout=interval_sec):
        if _try_init_once(ray=ray, torchrun=torchrun):
            _fire_on_ready()
            return


def _start(
    interval_sec: float = 2.0,
    *,
    ray: bool = True,
    torchrun: bool = True,
) -> None:
    global _bootstrap_thread
    if _bootstrap_thread is not None and _bootstrap_thread.is_alive():
        return
    _stop.clear()
    _bootstrap_thread = threading.Thread(
        target=_bootstrap_loop,
        args=(interval_sec,),
        kwargs={"ray": ray, "torchrun": torchrun},
        name="pulsing-bootstrap",
        daemon=True,
    )
    _bootstrap_thread.start()
    logger.debug(
        "Pulsing bootstrap thread started (ray=%s, torchrun=%s)", ray, torchrun
    )


def bootstrap(
    *,
    ray: bool = True,
    torchrun: bool = True,
    on_ready: Callable[..., None] | None = None,
    interval_sec: float = 2.0,
    wait_timeout: float | None = None,
) -> bool | None:
    """
    Start auto cluster formation in a Ray or torchrun environment.

    Use init(addr=..., seeds=...) (or init() for standalone) when you have explicit
    config; use bootstrap() only when this process is inside Ray/torchrun and you
    want Pulsing to auto-join. If the system is already initialized (e.g. you
    called init() earlier), bootstrap() only runs on_ready if provided.

    - ray: if True, try init_in_ray (default True). 不传则与 torchrun 一起默认 True，即自动检测（两个都试）。
    - torchrun: if True, try init_in_torchrun (default True).
    - bootstrap() — 不传 ray/torchrun 时默认两个都试，相当于自动检测环境；start background only.
    - bootstrap(ray=False, torchrun=True) — only try torchrun.
    - bootstrap(on_ready=callback) — call callback when cluster is ready.
      Callback can be () -> None or (system: ActorSystem) -> None.
    - bootstrap(wait_timeout=30) — block until ready or timeout; returns True/False.

    Returns True if wait_timeout was set and cluster became ready in time;
    False if wait_timeout was set and timed out; None if wait_timeout was not set.
    """
    from pulsing.core import is_initialized

    if on_ready is not None:
        with _lock:
            _on_ready_callbacks.append(on_ready)
        if is_initialized():
            _fire_on_ready()

    _start(interval_sec, ray=ray, torchrun=torchrun)

    if wait_timeout is None:
        return None

    deadline = time.monotonic() + wait_timeout
    while True:
        if is_initialized():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


def stop() -> None:
    """Stop the bootstrap background loop."""
    _stop.set()


__all__ = ["bootstrap", "stop"]
