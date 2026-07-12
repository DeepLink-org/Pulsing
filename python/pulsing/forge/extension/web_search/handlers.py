# SPDX-License-Identifier: Apache-2.0
"""Hosted ``web_search`` — Provider-side tool (Responses API), not sandbox execution."""

from __future__ import annotations

import json
import re
from typing import Any

from pulsing.forge.context import ToolCallContext
from pulsing.forge.result import ToolResult

# Codex registers web_search as a Responses API tool ({type: "web_search"}), not a
# sandbox function. Forge records calls; Craft enables provider execution.
_DEFERRED_MESSAGE = (
    "web_search is a hosted Provider tool (Responses API type=web_search). "
    "Forge recorded this call only; enable web_search in Craft's model config so the "
    "provider executes it. For client-side fetch/search, use web.run "
    "(FORGE_WEB_ALLOW / FORGE_WEB_SEARCH)."
)
_NO_PROVIDER_AUTH_MESSAGE = (
    "web_search provider credentials are not configured. "
    "Set the provider API key in Craft (e.g. OPENAI_API_KEY) and enable web_search "
    "in the model tools list. Forge does not execute hosted search locally."
)
_AUTH_ERROR_HINT = re.compile(
    r"(api[_ -]?key|authentication|unauthorized|invalid[_ -]?key|credentials?|"
    r"permission denied|401|403|no key)",
    re.IGNORECASE,
)


def _clean_query(query: Any) -> str:
    return str(query).strip() if query else ""


def _clean_queries(queries: Any) -> list[str]:
    if not isinstance(queries, list):
        return []
    return [s for q in queries if (s := str(q).strip())]


def _json_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        structured=payload,
        is_error=is_error,
    )


def _deferred_stub(arguments: dict[str, Any]) -> ToolResult:
    structured = {
        "kind": "hosted_web_search",
        "status": "deferred",
        "reason": "provider_not_configured",
        "executed_by": "provider",
        "arguments": arguments,
        "message": _DEFERRED_MESSAGE,
    }
    return _json_result(structured)


def _format_hook_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if _AUTH_ERROR_HINT.search(text):
        return f"web_search provider auth missing: {text}. {_NO_PROVIDER_AUTH_MESSAGE}"
    return f"web_search provider hook failed: {text}"


def _normalize_hook_result(out: Any, *, arguments: dict[str, Any]) -> ToolResult:
    if out is None:
        return _deferred_stub(arguments)
    if isinstance(out, ToolResult):
        return out
    if not isinstance(out, dict):
        return ToolResult(content=str(out))

    if out.get("is_error") or out.get("status") in {"error", "failed"}:
        message = str(
            out.get("message") or out.get("error") or out.get("content") or out
        )
        if _AUTH_ERROR_HINT.search(message):
            message = f"web_search provider auth missing: {message}. {_NO_PROVIDER_AUTH_MESSAGE}"
        return _json_result({**out, "message": message}, is_error=True)

    if out.get("status") == "deferred":
        return _deferred_stub(arguments)

    return _json_result(out)


def handle_web_search(
    *,
    ctx: ToolCallContext,
    query: Any = "",
    queries: Any = None,
    **kwargs: Any,
) -> ToolResult:
    """Forge records the call; actual search runs on the LLM provider when enabled in Craft."""
    query = _clean_query(query)
    queries = _clean_queries(queries)
    if not query and not queries:
        return ToolResult(
            content="web_search requires a non-empty 'query' or 'queries' argument",
            is_error=True,
        )

    arguments: dict[str, Any] = dict(kwargs)
    if query:
        arguments["query"] = query
    if queries:
        arguments["queries"] = queries

    hook = getattr(ctx.session, "hosted_web_search", None)
    if not callable(hook):
        return _deferred_stub(arguments)

    try:
        out = hook(arguments)
    except Exception as exc:
        return ToolResult(content=_format_hook_error(exc), is_error=True)

    return _normalize_hook_result(out, arguments=arguments)
