# SPDX-License-Identifier: Apache-2.0
"""Codex plugin compatibility layer (re-exports + legacy scan helpers)."""

from __future__ import annotations

from pathlib import Path

from pulsing.forge.discovery.codex_manifest import (
    find_plugin_manifest_path,
    load_codex_manifest,
)
from pulsing.forge.discovery.codex_paths import (
    plugins_cache_root,
    scan_codex_plugin_dirs_legacy,
)
from pulsing.forge.discovery.discoverable import (
    DiscoverablePlugin,
    DiscoverableToolType,
)
from pulsing.forge.discovery.plugin_id import PluginId
from pulsing.forge.discovery.plugin_store import PluginStore


def scan_codex_plugin_dirs(extra_dirs: list[str] | None = None) -> list[Path]:
    return scan_codex_plugin_dirs_legacy(extra_dirs)


def load_installed_manifests() -> list[tuple[object, Path]]:
    """Load manifests from Codex cache layout."""
    store = PluginStore()
    out: list[tuple[object, Path]] = []
    for pid in store.list_installed_plugin_ids():
        root = store.active_plugin_root(pid)
        if root is None:
            continue
        manifest_path = find_plugin_manifest_path(root)
        if manifest_path is None:
            continue
        out.append((load_codex_manifest(root), manifest_path))
    return out


# Back-compat re-export
__all__ = [
    "DiscoverablePlugin",
    "DiscoverableToolType",
    "PluginId",
    "load_installed_manifests",
    "scan_codex_plugin_dirs",
]
