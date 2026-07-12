# SPDX-License-Identifier: Apache-2.0
"""request_permissions approval bridge and checker tests."""

from __future__ import annotations

import pytest

from pulsing.forge.approval_bridge import make_worker_permissions_callback
from pulsing.forge.permissions import (
    PermissionChecker,
    is_permission_profile_effectively_empty,
    is_permission_section_empty,
)
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE


def test_is_permission_section_empty_codex_shapes() -> None:
    assert is_permission_section_empty({"enabled": None})
    assert is_permission_section_empty({"entries": []})
    assert is_permission_section_empty({"read": [], "write": []})
    assert not is_permission_section_empty({"enabled": True})
    assert not is_permission_section_empty({"write": ["subdir"]})


def test_is_permission_profile_effectively_empty() -> None:
    assert is_permission_profile_effectively_empty({})
    assert is_permission_profile_effectively_empty(
        {"network": {"enabled": None}, "file_system": {"entries": []}}
    )
    assert not is_permission_profile_effectively_empty({"network": {"enabled": True}})


def test_worker_permissions_callback_requires_sink() -> None:
    cb = make_worker_permissions_callback(None)
    with pytest.raises(RuntimeError, match="approval sink"):
        cb({"permissions": {"network": {"enabled": True}}})


def test_worker_permissions_callback_rejects_empty_grant() -> None:
    cb = make_worker_permissions_callback("fake-sink")

    def _fake_ask(_sink: str, _args: dict) -> dict:
        return {"permissions": {"network": {"enabled": None}}, "scope": "turn"}

    import pulsing.forge.approval_bridge as bridge

    bridge.ask_request_permissions_sync = _fake_ask  # type: ignore[assignment]
    bridge.tell_forge_event_sync = lambda *_a, **_k: None  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="denied by host"):
        cb({"permissions": {"network": {"enabled": True}}})


def test_permission_checker_denies_request_permissions_without_callback() -> None:
    checker = PermissionChecker(auto_approve=False)
    out = checker.prompt_request_permissions(
        {"permissions": {"network": {"enabled": True}}}
    )
    assert out["permissions"] == {}
    assert out["strict_auto_review"] is False


def test_permission_checker_prompt_callback_grants() -> None:
    checker = PermissionChecker(
        prompt_callback=lambda _tool, _summary: "allow",
    )
    perms = {"network": {"enabled": True}}
    out = checker.prompt_request_permissions({"permissions": perms})
    assert out["permissions"] == perms
    assert out["scope"] == "session"
    assert out["strict_auto_review"] is True


def test_permission_checker_auto_approves_request_permissions() -> None:
    checker = PermissionChecker(auto_approve=True)
    perms = {"network": {"enabled": True}}
    out = checker.prompt_request_permissions({"permissions": perms})
    assert out["permissions"] == perms
    assert out["scope"] == "session"


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_rust_rejects_empty_permissions_request() -> None:
    from pulsing.forge.rust_runtime import RustForgeAdapter

    host = RustForgeAdapter.create(
        cwd=".",
        auto_approve=False,
        event_callback=lambda _e: None,
        request_permissions_callback=lambda _args: {
            "permissions": {},
            "scope": "turn",
            "strict_auto_review": False,
        },
    )
    out = host.call_tool(
        "request_permissions",
        {"permissions": {"network": {"enabled": True}}},
    )
    assert out.is_error
    assert "denied" in out.content.lower()


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_rust_rejects_empty_nested_permissions() -> None:
    from pulsing.forge.rust_runtime import RustForgeAdapter

    host = RustForgeAdapter.create(cwd=".", auto_approve=False)
    out = host.call_tool("request_permissions", {"permissions": {"network": {}}})
    assert out.is_error


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_rust_rejects_path_outside_cwd() -> None:
    from pulsing.forge.rust_runtime import RustForgeAdapter

    host = RustForgeAdapter.create(cwd=".", auto_approve=False)
    out = host.call_tool(
        "request_permissions",
        {"permissions": {"file_system": {"write": ["/etc/passwd"]}}},
    )
    assert out.is_error
    assert "outside working directory" in out.content.lower()
