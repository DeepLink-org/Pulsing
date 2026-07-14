# SPDX-License-Identifier: Apache-2.0
"""apply_patch path escape protection."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.patch_invocation import ParsedPatch, apply_parsed_patch
from pulsing.testing.forge_harness import local_runtime

pytestmark = pytest.mark.forge


def test_apply_patch_rejects_relative_escape_outside_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    patch = "*** Begin Patch\n*** Add File: ../escape.txt\n+pwned\n*** End Patch\n"
    rt = local_runtime(workspace)
    out = rt.call_tool("apply_patch", {"patch": patch})
    assert out.is_error
    assert "outside working directory" in out.content
    assert not (tmp_path / "escape.txt").exists()


def test_apply_patch_rejects_absolute_path_outside_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    patch = f"*** Begin Patch\n*** Add File: {outside}\n+pwned\n*** End Patch\n"
    rt = local_runtime(workspace)
    out = rt.call_tool("apply_patch", {"patch": patch})
    assert out.is_error
    assert "outside working directory" in out.content
    assert not outside.exists()


def test_apply_patch_allows_absolute_path_inside_cwd(tmp_path: Path) -> None:
    target = tmp_path / "abs.txt"
    patch = f"*** Begin Patch\n*** Add File: {target}\n+ok\n*** End Patch\n"
    rt = local_runtime(tmp_path)
    out = rt.call_tool("apply_patch", {"patch": patch})
    assert not out.is_error
    assert target.read_text(encoding="utf-8") == "ok\n"


def test_apply_patch_rejects_workdir_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    patch = "*** Begin Patch\n*** Add File: ok.txt\n+ok\n*** End Patch\n"
    with pytest.raises(ValueError, match="outside working directory"):
        apply_parsed_patch(ParsedPatch(patch=patch, workdir=".."), workspace)


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlinks unsupported")
def test_apply_patch_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside)
    patch = "*** Begin Patch\n*** Add File: link/pwned.txt\n+escaped\n*** End Patch\n"
    rt = local_runtime(workspace)
    out = rt.call_tool("apply_patch", {"patch": patch})
    assert out.is_error
    assert "outside working directory" in out.content
    assert not (outside / "pwned.txt").exists()
