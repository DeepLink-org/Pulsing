# SPDX-License-Identifier: Apache-2.0
"""Permission checks: read-only, auto-approve, exec approval (Codex-aligned)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from pulsing.agent.loop.tool_base import Tool

PermissionDecision = Literal["allow", "deny"]
CallbackDecision = Literal["allow", "deny", "once"]
ExecApprovalDecision = Literal[
    "approved",
    "denied",
    "approved_for_session",
    "approved_with_amendment",
    "abort",
]


class PermissionChecker:
    """Read-only tools auto-allow; mutating tools and shell exec need approval."""

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        prompt_callback: Callable[[str, str], CallbackDecision] | None = None,
        user_input_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        plugin_install_callback: Callable[[dict[str, Any]], bool | str] | None = None,
        exec_approval_callback: (
            Callable[[dict[str, Any]], ExecApprovalDecision] | None
        ) = None,
        permissions_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._auto_approve = auto_approve
        self._prompt_callback = prompt_callback
        self._user_input_callback = user_input_callback
        self._plugin_install_callback = plugin_install_callback
        self._exec_approval_callback = exec_approval_callback
        self._permissions_callback = permissions_callback
        self._always_allow: set[str] = set()

    @property
    def auto_approve(self) -> bool:
        return self._auto_approve

    def check(self, tool: Tool, inputs: dict) -> PermissionDecision:
        if tool.is_read_only():
            return "allow"
        if self._auto_approve:
            return "allow"
        if tool.name in self._always_allow:
            return "allow"

        if self._prompt_callback is None:
            return "deny"

        summary = json.dumps(inputs, ensure_ascii=False, default=str)[:4000]
        choice = self._prompt_callback(tool.name, summary)
        if choice == "allow":
            self._always_allow.add(tool.name)
            return "allow"
        if choice == "once":
            return "allow"
        return "deny"

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

    def prompt_user_input(self, args: dict[str, Any]) -> dict[str, Any]:
        from pulsing.forge.session_input import (
            resolve_user_input,
            validate_request_user_input,
        )

        validated = validate_request_user_input(args)
        return resolve_user_input(
            validated,
            auto_approve=self._auto_approve,
            user_input_callback=self._user_input_callback,
            prompt_callback=self._prompt_callback,
        )

    def prompt_plugin_install(self, args: dict[str, Any]) -> bool:
        if self._auto_approve:
            return True
        if self._plugin_install_callback is not None:
            out = self._plugin_install_callback(args)
            if isinstance(out, bool):
                return out
            return str(out).strip().lower() in (
                "approved",
                "allow",
                "once",
                "yes",
                "true",
            )
        if self._prompt_callback is not None:
            name = str(args.get("tool_name") or args.get("tool_id") or "plugin")
            reason = str(args.get("suggest_reason") or "")
            summary = f"Install plugin {name}?\n{reason}".strip()
            choice = self._prompt_callback("request_plugin_install", summary)
            return choice in ("allow", "once")
        return False
