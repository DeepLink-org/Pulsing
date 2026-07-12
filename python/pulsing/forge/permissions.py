# SPDX-License-Identifier: Apache-2.0
"""Forge permission / exec-approval helpers (Codex-aligned)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

ExecApprovalDecision = Literal[
    "approved",
    "denied",
    "approved_for_session",
    "approved_with_amendment",
    "abort",
]

CallbackDecision = Literal["allow", "deny", "once"]


def is_permission_section_empty(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    if not value:
        return True
    if len(value) == 1 and "enabled" in value and value.get("enabled") is None:
        return True
    entries = value.get("entries")
    if isinstance(entries, list) and not entries:
        return True
    read = value.get("read")
    write = value.get("write")
    read_empty = not isinstance(read, list) or not read
    write_empty = not isinstance(write, list) or not write
    if "read" in value or "write" in value:
        return read_empty and write_empty
    return False


def is_permission_profile_effectively_empty(perms: dict[str, Any] | None) -> bool:
    if not perms:
        return True
    net = perms.get("network")
    fs = perms.get("file_system")
    net_empty = net is None or is_permission_section_empty(net)
    fs_empty = fs is None or is_permission_section_empty(fs)
    return net_empty and fs_empty


class PermissionChecker:
    """Shell exec approval for Forge host/worker bridges."""

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        prompt_callback: Callable[[str, str], CallbackDecision] | None = None,
        exec_approval_callback: (
            Callable[[dict[str, Any]], ExecApprovalDecision] | None
        ) = None,
        permissions_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._auto_approve = auto_approve
        self._prompt_callback = prompt_callback
        self._exec_approval_callback = exec_approval_callback
        self._permissions_callback = permissions_callback

    @property
    def auto_approve(self) -> bool:
        return self._auto_approve

    def prompt_exec_approval(self, request: dict[str, Any]) -> ExecApprovalDecision:
        if self._auto_approve:
            return "approved"
        if self._exec_approval_callback is not None:
            return self._exec_approval_callback(request)
        if self._prompt_callback is not None:
            cmd = " ".join(str(x) for x in (request.get("command") or []))
            reason = str(request.get("reason") or request.get("justification") or "")
            summary = f"shell exec: {cmd}\n{reason}".strip()
            choice = self._prompt_callback("shell_command", summary)
            if choice == "allow":
                return "approved_with_amendment"
            if choice == "once":
                return "approved"
            return "denied"
        return "denied"

    def prompt_request_permissions(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._auto_approve:
            perms = args.get("permissions") or {}
            return {
                "permissions": perms,
                "scope": "session",
                "strict_auto_review": False,
            }
        if self._permissions_callback is not None:
            return self._permissions_callback(args)
        if self._prompt_callback is not None:
            summary = json.dumps(args, ensure_ascii=False, default=str)[:4000]
            choice = self._prompt_callback("request_permissions", summary)
            if choice in ("allow", "once"):
                return {
                    "permissions": args.get("permissions") or {},
                    "scope": "session",
                    "strict_auto_review": choice == "allow",
                }
        return {"permissions": {}, "scope": "turn", "strict_auto_review": False}
