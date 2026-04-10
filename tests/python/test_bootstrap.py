"""Tests for pulsing.bootstrap one-shot behavior."""

from __future__ import annotations

import importlib

import pytest

import pulsing.core as core_module

bootstrap_module = importlib.import_module("pulsing.bootstrap")


@pytest.fixture(autouse=True)
def reset_bootstrap_state():
    bootstrap_module.stop()
    bootstrap_module._bootstrap_thread = None
    bootstrap_module._on_ready_callbacks.clear()
    yield
    bootstrap_module.stop()
    bootstrap_module._bootstrap_thread = None
    bootstrap_module._on_ready_callbacks.clear()


def _assert_no_background_start(monkeypatch) -> list[bool]:
    started: list[bool] = []
    monkeypatch.setattr(
        bootstrap_module,
        "_start",
        lambda *args, **kwargs: started.append(True),
    )
    return started


def test_bootstrap_wait_timeout_zero_returns_true_and_fires_on_ready(monkeypatch):
    state = {"initialized": False}
    system = object()

    monkeypatch.setattr(core_module, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(core_module, "get_system", lambda: system)

    def fake_try_init_once(*, ray: bool, torchrun: bool) -> bool:
        state["initialized"] = True
        return True

    monkeypatch.setattr(bootstrap_module, "_try_init_once", fake_try_init_once)
    started = _assert_no_background_start(monkeypatch)

    seen: list[object] = []
    result = bootstrap_module.bootstrap(
        wait_timeout=0, on_ready=lambda ready_system: seen.append(ready_system)
    )

    assert result is True
    assert seen == [system]
    assert started == []


def test_bootstrap_wait_timeout_zero_returns_false_without_background_retry(
    monkeypatch,
):
    monkeypatch.setattr(core_module, "is_initialized", lambda: False)
    monkeypatch.setattr(bootstrap_module, "_try_init_once", lambda **kwargs: False)
    started = _assert_no_background_start(monkeypatch)

    result = bootstrap_module.bootstrap(wait_timeout=0)

    assert result is False
    assert started == []


def test_bootstrap_wait_timeout_zero_returns_true_when_already_initialized(
    monkeypatch,
):
    system = object()

    monkeypatch.setattr(core_module, "is_initialized", lambda: True)
    monkeypatch.setattr(core_module, "get_system", lambda: system)
    monkeypatch.setattr(
        bootstrap_module,
        "_try_init_once",
        lambda **kwargs: pytest.fail("_try_init_once should not run"),
    )
    started = _assert_no_background_start(monkeypatch)

    seen: list[object] = []
    result = bootstrap_module.bootstrap(
        wait_timeout=0, on_ready=lambda ready_system: seen.append(ready_system)
    )

    assert result is True
    assert seen == [system]
    assert started == []
