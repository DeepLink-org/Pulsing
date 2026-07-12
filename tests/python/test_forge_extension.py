# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for Forge Extension tools (web.run, skills, memories, web_search)."""

from __future__ import annotations

from pathlib import Path

from pulsing.forge.context import ToolCallContext
from pulsing.forge.handlers import dispatch_tool


def _ctx(tmp_path: Path) -> ToolCallContext:
    return ToolCallContext(cwd=tmp_path)


def test_skills_list_and_read(tmp_path: Path, monkeypatch) -> None:
    skills_root = tmp_path / ".agents" / "skills" / "demo-skill"
    skills_root.mkdir(parents=True)
    (skills_root / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill for tests\n---\n# Demo\nbody\n",
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path)
    listed = dispatch_tool("skills.list", {}, ctx=ctx)
    assert not listed.is_error
    assert "demo-skill" in listed.content

    read = dispatch_tool("skills.read", {"name": "demo-skill"}, ctx=ctx)
    assert not read.is_error
    assert "Demo skill for tests" in read.content


def test_memories_crud(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(tmp_path / "memories"))
    ctx = _ctx(tmp_path)
    add = dispatch_tool(
        "memories.add_ad_hoc_note",
        {
            "filename": "2026-05-23T12-00-00-remember-this.md",
            "note": "remember this detail",
        },
        ctx=ctx,
    )
    assert not add.is_error

    listed = dispatch_tool("memories.list", {}, ctx=ctx)
    assert not listed.is_error
    assert listed.structured is not None
    assert listed.structured.get("entries")

    search = dispatch_tool("memories.search", {"queries": ["remember"]}, ctx=ctx)
    assert not search.is_error
    assert search.structured is not None
    assert search.structured.get("matches")


def test_web_run_open_allowlist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")
    ctx = _ctx(tmp_path)
    out = dispatch_tool(
        "web.run",
        {"open": [{"ref_id": "https://example.com/"}]},
        ctx=ctx,
    )
    # Network may fail in CI; accept success or HTTP error, not allowlist error.
    assert "is not in FORGE_WEB_ALLOW" not in out.content


def test_web_search_hosted_stub(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = dispatch_tool("web_search", {"query": "pulsing actor"}, ctx=ctx)
    assert not out.is_error
    assert out.structured is not None
    assert out.structured.get("kind") == "hosted_web_search"
    assert out.structured.get("status") == "deferred"
    assert out.structured.get("reason") == "provider_not_configured"
