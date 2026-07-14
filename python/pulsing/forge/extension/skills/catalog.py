# SPDX-License-Identifier: Apache-2.0
"""Discover Agent Skills (SKILL.md) under Codex-standard paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pulsing.forge.extension.paths import agents_skills_roots

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_NAME_RE = re.compile(r"^name:\s*(.+)\s*$", re.MULTILINE)
_DESC_RE = re.compile(r"^description:\s*(.+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    path: str
    root: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "root": self.root,
        }


def _parse_skill_md(path: Path, root: Path) -> SkillEntry | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    name = path.parent.name
    description = ""
    body = text
    match = _FRONTMATTER.match(text)
    if match:
        meta = match.group(1)
        body = text[match.end() :]
        if m := _NAME_RE.search(meta):
            name = m.group(1).strip().strip('"').strip("'")
        if m := _DESC_RE.search(meta):
            description = m.group(1).strip().strip('"').strip("'")
    if not description:
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("#"):
                description = line.lstrip("#").strip()
                break
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        return None
    return SkillEntry(name=name, description=description, path=rel, root=str(root))


def _resolved_within_root(path: Path, root: Path) -> bool:
    try:
        safe_root = root.resolve()
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == safe_root or safe_root in resolved.parents


def _has_symlink_component(root: Path, path: Path) -> bool:
    """True when any path component under ``root`` is a symlink (lstat, no follow)."""
    try:
        safe_root = root.resolve()
        rel = path.relative_to(safe_root)
    except (OSError, ValueError):
        return True
    candidate = safe_root
    for part in rel.parts:
        if part in ("..", ""):
            return True
        candidate = candidate / part
        if candidate.is_symlink():
            return True
    return False


def _skill_md_is_listable(skill_md: Path, safe_root: Path) -> bool:
    # Symlinked SKILL.md leaves still match a filename walk and would leak
    # arbitrary file content via frontmatter/body parsing if read.
    if skill_md.is_symlink() or not skill_md.is_file():
        return False
    if _has_symlink_component(safe_root, skill_md):
        return False
    return _resolved_within_root(skill_md, safe_root)


def _iter_skill_md_files(safe_root: Path) -> list[Path]:
    """Collect SKILL.md files under ``safe_root`` without following symlink dirs."""
    found: list[Path] = []
    stack = [safe_root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for child in children:
            if child.is_symlink():
                continue
            if child.name == "SKILL.md" and child.is_file():
                found.append(child)
            elif child.is_dir():
                stack.append(child)
    return found


def list_skills(cwd: Path) -> list[SkillEntry]:
    seen: set[str] = set()
    out: list[SkillEntry] = []
    for root in agents_skills_roots(cwd):
        try:
            safe_root = root.resolve()
            if not safe_root.is_dir():
                continue
            candidates = _iter_skill_md_files(safe_root)
        except OSError:
            continue
        for skill_md in candidates:
            if not _skill_md_is_listable(skill_md, safe_root):
                continue
            entry = _parse_skill_md(skill_md, safe_root)
            if entry is None:
                continue
            key = f"{entry.root}:{entry.path}"
            if key in seen:
                continue
            seen.add(key)
            out.append(entry)
    return out


_SKILL_READ_CAP = 2 * 1024 * 1024


def _resolve_within_root(root: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` under ``root``, rejecting any symlink hop or escape.

    ``entry.path`` values always come from :func:`list_skills`'s own directory
    walk, but we re-validate here (rather than trusting the cached entry) so a
    symlink swapped in between listing and reading cannot smuggle an out-of-root
    file into the response.
    """
    safe_root = root.resolve()
    candidate = safe_root
    for part in Path(relative_path).parts:
        if part in ("..", ""):
            raise FileNotFoundError(f"skill not found: {relative_path}")
        candidate = candidate / part
    if _has_symlink_component(safe_root, candidate):
        raise FileNotFoundError(f"skill not found: {relative_path}")
    resolved = candidate.resolve()
    if resolved != safe_root and safe_root not in resolved.parents:
        raise FileNotFoundError(f"skill not found: {relative_path}")
    return resolved


def _read_skill_file(entry: SkillEntry) -> str:
    file_path = _resolve_within_root(Path(entry.root), entry.path)
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(f"skill not found: {entry.path}") from exc
    if size > _SKILL_READ_CAP:
        raise OSError(f"skill file too large to read (max {_SKILL_READ_CAP} bytes)")
    return file_path.read_text(encoding="utf-8")


def read_skill(*, cwd: Path, name: str = "", path: str = "") -> str:
    target_name = name.strip()
    target_path = path.strip()
    for entry in list_skills(cwd):
        if target_path and entry.path == target_path:
            return _read_skill_file(entry)
        if target_name and entry.name == target_name:
            return _read_skill_file(entry)
    raise FileNotFoundError(f"skill not found: {name or path}")
