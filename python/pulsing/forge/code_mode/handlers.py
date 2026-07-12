# SPDX-License-Identifier: Apache-2.0
"""Forge tool handlers for Codex-aligned ``exec`` / ``wait``."""

from __future__ import annotations

from typing import Any

from pulsing.forge.context import ToolCallContext
from pulsing.forge.code_mode.protocol import PUBLIC_TOOL_NAME, WAIT_TOOL_NAME
from pulsing.forge.code_mode.tools_bridge import ToolsBridge
from pulsing.forge.result import ToolResult
from pulsing.forge.sandbox import normalize_policy


def _exec_source_from_args(source: str = "", **kwargs: Any) -> str:
    raw = (
        source
        or kwargs.get("input")
        or kwargs.get("code")
        or kwargs.get("source")
        or ""
    )
    return str(raw)


def _rejects_sandbox_policy(
    ctx: ToolCallContext,
    *,
    sandbox_policy: str | None,
    dangerously_disable_sandbox: bool,
) -> str | None:
    """``exec`` runs the cell in-process with no OS-level isolation.

    Unlike ``shell_command``/``exec_command``, there is no subprocess or
    namespace boundary to apply ``sandbox_policy`` to. Rather than silently
    ignoring a caller's request for isolation (a sandbox-boundary bypass),
    fail closed whenever isolation was explicitly requested.
    """
    if dangerously_disable_sandbox or ctx.dangerously_disable_sandbox:
        return None
    # Fail closed if *either* the session or per-call args request isolation.
    # Per-call sandbox_policy="off" must not downgrade a restricted session
    # (dispatch_tool uses setdefault, so model-supplied args can bypass ctx).
    policies = [normalize_policy(ctx.sandbox_policy)]
    if sandbox_policy is not None:
        policies.append(normalize_policy(sandbox_policy))
    for policy in policies:
        if policy != "off":
            return (
                f"exec runs Python in-process and cannot honor sandbox_policy={policy!r} "
                "(no filesystem/network isolation is available); set sandbox_policy=off "
                "on the session and omit per-call overrides, or pass "
                "dangerously_disable_sandbox=true to proceed at your own risk"
            )
    return None


def handle_exec(
    *,
    ctx: ToolCallContext,
    source: str = "",
    sandbox_policy: str | None = None,
    dangerously_disable_sandbox: bool = False,
    **kwargs: Any,
) -> ToolResult:
    from pulsing.forge.handlers import dispatch_tool

    text = _exec_source_from_args(source, **kwargs)
    if not text.strip():
        return ToolResult(content="exec source is empty", is_error=True)

    rejection = _rejects_sandbox_policy(
        ctx,
        sandbox_policy=sandbox_policy,
        dangerously_disable_sandbox=dangerously_disable_sandbox,
    )
    if rejection is not None:
        return ToolResult(content=rejection, is_error=True)

    bridge = ToolsBridge(lambda name, args: dispatch_tool(name, args, ctx=ctx))
    try:
        response = ctx.code_mode.execute(text, bridge)
    except ValueError as exc:
        return ToolResult(content=str(exc), is_error=True)

    return ToolResult(
        content=response.model_message(),
        structured=response.to_dict(),
    )


def handle_wait(*, ctx: ToolCallContext, **kwargs: Any) -> ToolResult:
    from pulsing.forge.code_mode.protocol import WaitArgs

    try:
        args = WaitArgs.from_dict(dict(kwargs))
    except (TypeError, ValueError) as exc:
        return ToolResult(content=str(exc), is_error=True)
    if not args.cell_id:
        return ToolResult(content="cell_id is required", is_error=True)

    response = ctx.code_mode.wait(args)
    if response.error_text and response.error_text.startswith("unknown cell_id:"):
        return ToolResult(content=response.error_text, is_error=True)

    return ToolResult(
        content=response.model_message(),
        structured=response.to_dict(),
    )


CODE_MODE_TOOL_NAMES: frozenset[str] = frozenset({PUBLIC_TOOL_NAME, WAIT_TOOL_NAME})
