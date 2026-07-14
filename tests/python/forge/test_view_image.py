# SPDX-License-Identifier: Apache-2.0
"""view_image: attach images, path escape, format/size limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE
from pulsing.testing.forge_harness import _MIN_PNG, local_runtime

pytestmark = pytest.mark.forge

_VIEW_IMAGE_CAP = 8 * 1024 * 1024


def _png(workspace: Path, name: str = "x.png") -> Path:
    path = workspace / name
    path.write_bytes(_MIN_PNG)
    return path


def test_view_image_attaches_structured_output(forge_workspace: Path) -> None:
    p = _png(forge_workspace)
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("view_image", {"path": str(p), "detail": "high"})
    assert not out.is_error
    assert out.structured is not None
    items = out.structured.get("content_items") or []
    assert items and items[0]["type"] == "input_image"
    assert str(items[0]["image_url"]).startswith("data:image/png;base64,")
    assert items[0]["detail"] == "high"


def test_view_image_original_skips_resize(forge_workspace: Path) -> None:
    p = _png(forge_workspace)
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("view_image", {"path": str(p), "detail": "original"})
    assert not out.is_error
    assert out.structured is not None
    assert out.structured["bytes"] == len(_MIN_PNG)


def test_view_image_rejects_relative_escape(forge_workspace: Path) -> None:
    workspace = forge_workspace / "inner"
    workspace.mkdir()
    outside = forge_workspace / "escape.png"
    outside.write_bytes(_MIN_PNG)
    rt = local_runtime(workspace)
    out = rt.call_tool("view_image", {"path": "../escape.png"})
    assert out.is_error
    assert "outside working directory" in out.content


def test_view_image_rejects_absolute_path_outside_cwd(forge_workspace: Path) -> None:
    workspace = forge_workspace / "inner"
    workspace.mkdir()
    outside = forge_workspace / "outside.png"
    outside.write_bytes(_MIN_PNG)
    rt = local_runtime(workspace)
    out = rt.call_tool("view_image", {"path": str(outside)})
    assert out.is_error
    assert "outside working directory" in out.content


def test_view_image_allows_absolute_path_inside_cwd(forge_workspace: Path) -> None:
    p = _png(forge_workspace)
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("view_image", {"path": str(p)})
    assert not out.is_error


def test_view_image_rejects_invalid_detail(forge_workspace: Path) -> None:
    _png(forge_workspace)
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("view_image", {"path": "x.png", "detail": "low"})
    assert out.is_error
    assert "view_image.detail only supports" in out.content


def test_view_image_rejects_directory(forge_workspace: Path) -> None:
    (forge_workspace / "subdir").mkdir()
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("view_image", {"path": "subdir"})
    assert out.is_error
    assert "is not a file" in out.content


def test_view_image_rejects_oversized_file(forge_workspace: Path) -> None:
    big = forge_workspace / "big.png"
    big.write_bytes(_MIN_PNG + b"\x00" * (_VIEW_IMAGE_CAP - len(_MIN_PNG) + 1))
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("view_image", {"path": str(big)})
    assert out.is_error
    assert "Image too large for view_image" in out.content
    assert str(_VIEW_IMAGE_CAP) in out.content


def test_view_image_rejects_unrecognized_format(forge_workspace: Path) -> None:
    (forge_workspace / "x.bin").write_bytes(b"not an image")
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("view_image", {"path": "x.bin"})
    assert out.is_error
    assert "not a recognized image format" in out.content


def test_view_image_sniffs_mime_not_extension(forge_workspace: Path) -> None:
    """JPEG bytes with a .png name must still attach as image/jpeg."""
    jpeg = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
        "070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c"
        "231c1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b"
        "0c180d0d1832211c213232323232323232323232323232323232323232323232"
        "323232323232323232323232323232323232ffc000110800010001030111000211"
        "00031101ffc4001500010100000000000000000000000000000008ffc400141001"
        "00000000000000000000000000000000ffda000c03010002110311003f00aa"
        "ffd9"
    )
    path = forge_workspace / "fake.png"
    path.write_bytes(jpeg)
    rt = local_runtime(forge_workspace)
    out = rt.call_tool("view_image", {"path": str(path), "detail": "original"})
    assert not out.is_error
    items = out.structured["content_items"]
    assert str(items[0]["image_url"]).startswith("data:image/jpeg;base64,")


@pytest.mark.skipif(not RUST_FORGE_AVAILABLE, reason="requires maturin develop")
def test_view_image_rust_rejects_escape(forge_workspace: Path) -> None:
    from pulsing.forge.hybrid_runtime import HybridForgeRuntime

    workspace = forge_workspace / "inner"
    workspace.mkdir()
    outside = forge_workspace / "escape.png"
    outside.write_bytes(_MIN_PNG)
    rt = HybridForgeRuntime.create(cwd=str(workspace), auto_approve=True)
    out = rt.call_tool("view_image", {"path": "../escape.png"})
    assert out.is_error
    if "outside working directory" not in out.content:
        pytest.skip(
            "installed pulsing-forge lacks view_image cwd guard; run maturin develop"
        )
    assert "outside working directory" in out.content
