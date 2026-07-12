# SPDX-License-Identifier: Apache-2.0
"""Security-focused tests for the ``skills.read`` Forge tool.

Covers path-traversal and symlink-escape attempts against the skills
catalog, in addition to basic read/size-cap behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.extension.skills.catalog import (
    _SKILL_READ_CAP,
    list_skills,
    read_skill,
)
from pulsing.forge.handlers import dispatch_tool


def _ctx(cwd: Path) -> ToolCallContext:
    return ToolCallContext(cwd=cwd)


def _write_skill(root: Path, skill_name: str, *, name: str, description: str) -> Path:
    skill_dir = root / ".agents" / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\nbody\n",
        encoding="utf-8",
    )
    return skill_md


def _isolate_skills_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FORGE_SKILLS_DIRS", raising=False)


def test_skills_list_empty_when_skills_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_skills_home(monkeypatch, tmp_path)
    assert list_skills(tmp_path) == []
    listed = dispatch_tool("skills.list", {}, ctx=_ctx(tmp_path))
    assert not listed.is_error
    assert listed.structured == {"skills": []}


def test_skills_list_excludes_symlinked_skill_md(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET CONTENT", encoding="utf-8")
    skill_dir = tmp_path / ".agents" / "skills" / "evil-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").symlink_to(secret)

    listed = dispatch_tool("skills.list", {}, ctx=_ctx(tmp_path))
    assert not listed.is_error
    assert "evil-skill" not in listed.content
    assert "TOP SECRET" not in listed.content


def test_skills_list_excludes_symlinked_skill_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: outside-skill\ndescription: should not be listed\n---\n",
        encoding="utf-8",
    )
    skills_root = tmp_path / ".agents" / "skills"
    skills_root.mkdir(parents=True)
    (skills_root / "linked-skill").symlink_to(outside)

    listed = dispatch_tool("skills.list", {}, ctx=_ctx(tmp_path))
    assert not listed.is_error
    assert "outside-skill" not in listed.content
    assert "should not be listed" not in listed.content


def test_skills_list_skips_invalid_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_skills_home(monkeypatch, tmp_path)
    bad = tmp_path / ".agents" / "skills" / "bad-utf8"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_bytes(b"\xff\xfe not utf-8")
    good = tmp_path / ".agents" / "skills" / "good-skill"
    good.mkdir()
    (good / "SKILL.md").write_text(
        "---\nname: good-skill\ndescription: valid skill\n---\n",
        encoding="utf-8",
    )

    listed = dispatch_tool("skills.list", {}, ctx=_ctx(tmp_path))
    assert not listed.is_error
    assert "good-skill" in listed.content
    assert "bad-utf8" not in listed.content


def test_skills_read_by_name_and_path(tmp_path: Path) -> None:
    _write_skill(
        tmp_path, "demo-skill", name="demo-skill", description="Demo skill for tests"
    )
    ctx = _ctx(tmp_path)

    read = dispatch_tool("skills.read", {"name": "demo-skill"}, ctx=ctx)
    assert not read.is_error
    assert "Demo skill for tests" in read.content

    read_by_path = dispatch_tool(
        "skills.read", {"path": "demo-skill/SKILL.md"}, ctx=ctx
    )
    assert not read_by_path.is_error
    assert "Demo skill for tests" in read_by_path.content


def test_skills_read_missing_name_or_path_errors(tmp_path: Path) -> None:
    out = dispatch_tool("skills.read", {}, ctx=_ctx(tmp_path))
    assert out.is_error


def test_skills_read_unknown_skill_errors_without_leaking_paths(tmp_path: Path) -> None:
    out = dispatch_tool("skills.read", {"name": "does-not-exist"}, ctx=_ctx(tmp_path))
    assert out.is_error
    assert str(tmp_path) not in out.content


@pytest.mark.parametrize(
    "traversal_path",
    [
        "../../../etc/passwd",
        "demo-skill/../../../etc/passwd",
        "/etc/passwd",
    ],
)
def test_skills_read_rejects_path_traversal(
    tmp_path: Path, traversal_path: str
) -> None:
    _write_skill(tmp_path, "demo-skill", name="demo-skill", description="Demo skill")
    ctx = _ctx(tmp_path)

    out = dispatch_tool("skills.read", {"path": traversal_path}, ctx=ctx)
    assert out.is_error
    assert "root:" not in out.content
    assert str(tmp_path) not in out.content


def test_skills_read_rejects_symlinked_skill_file(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET CONTENT", encoding="utf-8")

    skill_dir = tmp_path / ".agents" / "skills" / "evil-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").symlink_to(secret)

    ctx = _ctx(tmp_path)
    out = dispatch_tool("skills.read", {"name": "evil-skill"}, ctx=ctx)
    assert out.is_error
    assert "TOP SECRET" not in out.content

    out_by_path = dispatch_tool("skills.read", {"path": "evil-skill/SKILL.md"}, ctx=ctx)
    assert out_by_path.is_error
    assert "TOP SECRET" not in out_by_path.content


def test_skills_read_rejects_symlinked_skill_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: outside-skill\ndescription: should not be reachable\n---\n",
        encoding="utf-8",
    )

    skills_root = tmp_path / ".agents" / "skills"
    skills_root.mkdir(parents=True)
    (skills_root / "linked-skill").symlink_to(outside)

    ctx = _ctx(tmp_path)
    out = dispatch_tool("skills.read", {"name": "outside-skill"}, ctx=ctx)
    assert out.is_error
    assert "should not be reachable" not in out.content


def test_read_skill_direct_call_rejects_escape(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET CONTENT", encoding="utf-8")
    skill_dir = tmp_path / ".agents" / "skills" / "evil-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").symlink_to(secret)

    with pytest.raises(FileNotFoundError):
        read_skill(cwd=tmp_path, name="evil-skill")


def test_skills_read_enforces_size_cap(tmp_path: Path) -> None:
    skill_md = _write_skill(
        tmp_path, "big-skill", name="big-skill", description="Big skill"
    )
    skill_md.write_text("x" * (_SKILL_READ_CAP + 1), encoding="utf-8")

    out = dispatch_tool("skills.read", {"name": "big-skill"}, ctx=_ctx(tmp_path))
    assert out.is_error
    assert "too large" in out.content
