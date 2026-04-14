"""Focused unit tests for pulsing._runtime branch behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pulsing
import pulsing._runtime as runtime


@pytest.fixture(autouse=True)
def reset_runtime_state(monkeypatch):
    monkeypatch.setattr(runtime, "_module_owns_system", False)
    monkeypatch.setattr(runtime, "_atexit_registered", False)
    monkeypatch.setattr(runtime, "_async_init_lock", None)
    monkeypatch.setattr(runtime, "_async_init_lock_loop", None)


def test_ensure_sync_runtime_requires_routable_addr_for_initialized_system(
    monkeypatch,
):
    monkeypatch.setattr(pulsing, "is_initialized", lambda: True)
    monkeypatch.setattr(pulsing, "get_system", lambda: SimpleNamespace(addr=None))

    with pytest.raises(RuntimeError, match="requires a routable Pulsing address"):
        runtime.ensure_sync_runtime(require_addr=True)


def test_ensure_sync_runtime_rejects_initialized_system_without_dispatch_loop(
    monkeypatch,
):
    monkeypatch.setattr(pulsing, "is_initialized", lambda: True)
    monkeypatch.setattr(
        pulsing,
        "get_system",
        lambda: SimpleNamespace(addr="127.0.0.1:12345"),
    )
    monkeypatch.setattr(runtime, "get_loop", lambda: None)

    with pytest.raises(RuntimeError, match="dispatch loop is unavailable"):
        runtime.ensure_sync_runtime()


def test_ensure_sync_runtime_rejects_bootstrap_without_dispatch_loop(monkeypatch):
    monkeypatch.setattr(pulsing, "is_initialized", lambda: False)
    monkeypatch.setattr(pulsing, "bootstrap", lambda wait_timeout=0: True)
    monkeypatch.setattr(runtime, "get_loop", lambda: None)

    with pytest.raises(RuntimeError, match="no running dispatch loop was exposed"):
        runtime.ensure_sync_runtime()


def test_ensure_sync_runtime_claims_bootstrap_initialized_system(monkeypatch):
    cleanup_calls = []

    monkeypatch.setattr(pulsing, "is_initialized", lambda: False)
    monkeypatch.setattr(pulsing, "bootstrap", lambda wait_timeout=0: True)
    monkeypatch.setattr(
        pulsing,
        "get_system",
        lambda: SimpleNamespace(addr="127.0.0.1:12345"),
    )
    monkeypatch.setattr(runtime, "get_loop", lambda: object())
    monkeypatch.setattr(
        runtime,
        "_register_cleanup",
        lambda: cleanup_calls.append("registered"),
    )

    runtime.ensure_sync_runtime(require_addr=True)

    assert runtime.owns_system() is True
    assert cleanup_calls == ["registered"]


def test_ensure_sync_runtime_preserves_existing_module_ownership(monkeypatch):
    monkeypatch.setattr(runtime, "_module_owns_system", True)
    monkeypatch.setattr(pulsing, "is_initialized", lambda: True)
    monkeypatch.setattr(
        pulsing,
        "get_system",
        lambda: SimpleNamespace(addr="127.0.0.1:12345"),
    )
    monkeypatch.setattr(runtime, "get_loop", lambda: object())

    runtime.ensure_sync_runtime(require_addr=True)

    assert runtime.owns_system() is True


def test_ensure_sync_runtime_clears_ownership_when_init_bridge_fails(monkeypatch):
    monkeypatch.setattr(pulsing, "is_initialized", lambda: False)
    monkeypatch.setattr(pulsing, "bootstrap", lambda wait_timeout=0: False)
    monkeypatch.setattr(
        pulsing, "init", lambda addr=None: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(runtime, "_register_cleanup", lambda: None)

    def fail_bridge(awaitable):
        awaitable.close()
        raise RuntimeError("bridge boom")

    monkeypatch.setattr(runtime, "bridge_run_sync", fail_bridge)

    with pytest.raises(RuntimeError, match="bridge boom"):
        runtime.ensure_sync_runtime()

    assert runtime.owns_system() is False


def test_shutdown_reraises_when_best_effort_disabled(monkeypatch):
    monkeypatch.setattr(runtime, "_module_owns_system", True)
    monkeypatch.setattr(pulsing, "is_initialized", lambda: True)
    monkeypatch.setattr(
        pulsing, "shutdown", lambda: SimpleNamespace(close=lambda: None)
    )

    def fail_bridge(awaitable):
        awaitable.close()
        raise RuntimeError("shutdown boom")

    monkeypatch.setattr(runtime, "bridge_run_sync", fail_bridge)

    with pytest.raises(RuntimeError, match="shutdown boom"):
        runtime.shutdown(best_effort=False)
