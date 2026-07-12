# SPDX-License-Identifier: Apache-2.0
"""Dedicated tests for tool_search (BM25, limits, handler parity with Rust)."""

from __future__ import annotations

import json

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.discovery.bm25 import bm25_scores
from pulsing.forge.discovery.catalog import ToolCatalog
from pulsing.forge.discovery.entries import (
    TOOL_SEARCH_DEFAULT_LIMIT,
    TOOL_SEARCH_MAX_LIMIT,
    TOOL_SEARCH_MAX_QUERY_CHARS,
    DeferredToolEntry,
    normalize_tool_search_limit,
)
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.session import LocalToolSession


def _catalog(*names: tuple[str, str]) -> ToolCatalog:
    catalog = ToolCatalog()
    for name, desc in names:
        catalog.register_deferred(DeferredToolEntry.from_function(name, desc))
    return catalog


def test_bm25_ranks_github_higher() -> None:
    docs = ["filesystem read write grep", "github pull request mcp server"]
    scores = bm25_scores("github mcp", docs)
    assert scores[1] > scores[0]


def test_bm25_empty_query_scores_zero() -> None:
    docs = ["github mcp server", "filesystem tools"]
    assert bm25_scores("", docs) == [0.0, 0.0]
    assert bm25_scores("   ", docs) == [0.0, 0.0]


def test_bm25_punctuation_only_query() -> None:
    docs = ["github mcp server"]
    assert bm25_scores("!!! ??? ---", docs) == [0.0]


def test_bm25_no_documents() -> None:
    assert bm25_scores("github", []) == []


def test_normalize_tool_search_limit() -> None:
    assert normalize_tool_search_limit(None) == TOOL_SEARCH_DEFAULT_LIMIT
    assert normalize_tool_search_limit(0) == TOOL_SEARCH_DEFAULT_LIMIT
    assert normalize_tool_search_limit(-5) == TOOL_SEARCH_DEFAULT_LIMIT
    assert normalize_tool_search_limit("bad") == TOOL_SEARCH_DEFAULT_LIMIT
    assert normalize_tool_search_limit(3) == 3
    assert normalize_tool_search_limit(999_999) == TOOL_SEARCH_MAX_LIMIT


def test_catalog_search_respects_limit() -> None:
    catalog = _catalog(
        ("alpha", "alpha tool"),
        ("beta", "beta tool"),
        ("gamma", "gamma tool"),
    )
    hits = catalog.search("tool", limit=2)
    assert len(hits) == 2


def test_handler_rejects_empty_query() -> None:
    ctx = ToolCallContext(
        cwd=".", session=LocalToolSession(), tool_catalog=ToolCatalog()
    )
    for args in [{"query": ""}, {"query": "   "}, {}]:
        out = dispatch_tool("tool_search", args, ctx=ctx)
        assert out.is_error
        assert out.content == "tool_search requires non-empty query"


def test_handler_returns_loadable_json() -> None:
    catalog = _catalog(("github_mcp", "GitHub MCP integration"))
    ctx = ToolCallContext(cwd=".", session=LocalToolSession(), tool_catalog=catalog)
    out = dispatch_tool("tool_search", {"query": "github mcp"}, ctx=ctx)
    assert not out.is_error
    payload = json.loads(out.content)
    tool = payload["tools"][0]
    assert tool["type"] == "function"
    assert tool["name"] == "github_mcp"
    assert tool["defer_loading"] is True


@pytest.mark.parametrize("limit", [0, -5])
def test_handler_limit_non_positive_uses_default(limit: int) -> None:
    catalog = _catalog(
        ("github_mcp", "GitHub integration"),
        ("github_issues", "GitHub integration"),
        ("github_actions", "GitHub integration"),
    )
    ctx = ToolCallContext(cwd=".", session=LocalToolSession(), tool_catalog=catalog)
    out = dispatch_tool("tool_search", {"query": "github", "limit": limit}, ctx=ctx)
    assert not out.is_error
    assert len(json.loads(out.content)["tools"]) == 3


def test_handler_huge_limit_clamped() -> None:
    catalog = _catalog(("github_mcp", "GitHub MCP integration"))
    ctx = ToolCallContext(cwd=".", session=LocalToolSession(), tool_catalog=catalog)
    out = dispatch_tool(
        "tool_search", {"query": "github", "limit": 999_999_999}, ctx=ctx
    )
    assert not out.is_error
    assert len(json.loads(out.content)["tools"]) == 1


def test_handler_truncates_overlong_query() -> None:
    catalog = _catalog(("github_mcp", "GitHub MCP integration"))
    ctx = ToolCallContext(cwd=".", session=LocalToolSession(), tool_catalog=catalog)
    query = "github " + "x" * TOOL_SEARCH_MAX_QUERY_CHARS
    out = dispatch_tool("tool_search", {"query": query}, ctx=ctx)
    assert not out.is_error
    assert len(json.loads(out.content)["tools"]) == 1
