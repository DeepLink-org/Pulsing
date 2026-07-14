# SPDX-License-Identifier: Apache-2.0
"""Forge session + discovery tests."""

from __future__ import annotations

import json

import pytest

from pulsing.forge.discovery.catalog import DeferredToolEntry, ToolCatalog
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.context import ToolCallContext
from pulsing.forge.session import LocalToolSession
from pulsing.forge.session_input import validate_request_user_input


def test_validate_request_user_input_rejects_empty():
    with pytest.raises(ValueError):
        validate_request_user_input({"questions": []})


def test_tool_search_bm25():
    catalog = ToolCatalog()
    catalog.register_deferred(
        DeferredToolEntry.from_function("github_pr", "Open GitHub pull requests")
    )
    catalog.register_deferred(
        DeferredToolEntry.from_function("read_file", "Read workspace files")
    )
    hits = catalog.search("github pull request")
    assert hits[0].name == "github_pr"


def test_get_context_remaining_structured():
    session = LocalToolSession(token_budget=42_000)
    ctx = ToolCallContext(cwd=".", session=session)
    out = dispatch_tool("get_context_remaining", {}, ctx=ctx)
    payload = json.loads(out.content)
    assert payload["tokens_remaining"] == 42_000
    assert payload["status"] == "ok"


def test_list_plugins_empty_catalog():
    ctx = ToolCallContext(cwd=".", session=LocalToolSession())
    out = dispatch_tool("list_available_plugins_to_install", {}, ctx=ctx)
    payload = json.loads(out.content)
    assert "tools" in payload


def test_auto_resolution_timeout_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import pulsing.forge.session_input as session_input

    monkeypatch.setattr(session_input, "MIN_AUTO_RESOLUTION_MS", 1)

    args = session_input.validate_request_user_input(
        {
            "questions": [
                {
                    "id": "pick",
                    "header": "Go",
                    "question": "Which?",
                    "options": [
                        {"label": "A (Recommended)", "description": "first"},
                        {"label": "B", "description": "second"},
                    ],
                }
            ],
            "autoResolutionMs": 50,
        }
    )
    out = session_input.resolve_user_input(
        args, auto_approve=False, user_input_callback=None, prompt_callback=None
    )
    assert out["answers"]["pick"]["answers"] == ["A (Recommended)"]
