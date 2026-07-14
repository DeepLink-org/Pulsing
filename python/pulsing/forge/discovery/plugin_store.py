# SPDX-License-Identifier: Apache-2.0
"""Codex plugin cache store: ~/.codex/plugins/cache/{marketplace}/{name}/{version}/."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from pulsing.forge.discovery.codex_manifest import (
    find_plugin_manifest_path,
    load_codex_manifest,
)
from pulsing.forge.discovery.codex_paths import (
    DEFAULT_PLUGIN_VERSION,
    plugins_cache_root,
)
from pulsing.forge.discovery.marketplace import copy_tree_atomic
from pulsing.forge.discovery.plugin_id import PluginId


@dataclass
class PluginInstallResult:
    plugin_id: PluginId
    plugin_version: str
    installed_path: Path


class PluginStore:
    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = (cache_root or plugins_cache_root()).resolve()

    def plugin_base_root(self, plugin_id: PluginId) -> Path:
        return self.cache_root / plugin_id.marketplace_name / plugin_id.plugin_name

    def plugin_version_root(self, plugin_id: PluginId, version: str) -> Path:
        return self.plugin_base_root(plugin_id) / version

    def active_plugin_version(self, plugin_id: PluginId) -> str | None:
        base = self.plugin_base_root(plugin_id)
        if not base.is_dir():
            return None
        versions = [
            p.name
            for p in base.iterdir()
            if p.is_dir() and _valid_version_segment(p.name)
        ]
        if not versions:
            return None
        if DEFAULT_PLUGIN_VERSION in versions:
            return DEFAULT_PLUGIN_VERSION
        return sorted(versions, key=_version_sort_key)[-1]

    def active_plugin_root(self, plugin_id: PluginId) -> Path | None:
        version = self.active_plugin_version(plugin_id)
        if version is None:
            return None
        return self.plugin_version_root(plugin_id, version)

    def is_installed(self, plugin_id: PluginId) -> bool:
        root = self.active_plugin_root(plugin_id)
        return root is not None and find_plugin_manifest_path(root) is not None

    def install(self, source_path: Path, plugin_id: PluginId) -> PluginInstallResult:
        source_path = source_path.resolve()
        if not source_path.is_dir():
            raise ValueError(f"plugin source is not a directory: {source_path}")
        manifest = load_codex_manifest(source_path)
        if manifest.name != plugin_id.plugin_name:
            raise ValueError(
                f"plugin.json name {manifest.name!r} does not match marketplace name {plugin_id.plugin_name!r}"
            )
        version = manifest.version or DEFAULT_PLUGIN_VERSION
        _validate_version_segment(version)
        installed_path = self.plugin_version_root(plugin_id, version)
        copy_tree_atomic(source_path, installed_path)
        return PluginInstallResult(
            plugin_id=plugin_id,
            plugin_version=version,
            installed_path=installed_path,
        )

    def uninstall(self, plugin_id: PluginId) -> None:
        base = self.plugin_base_root(plugin_id)
        if base.exists():
            shutil.rmtree(base)

    def list_installed_plugin_ids(self) -> list[PluginId]:
        out: list[PluginId] = []
        if not self.cache_root.is_dir():
            return out
        for marketplace_dir in self.cache_root.iterdir():
            if not marketplace_dir.is_dir():
                continue
            for plugin_dir in marketplace_dir.iterdir():
                if not plugin_dir.is_dir():
                    continue
                pid = PluginId(
                    plugin_name=plugin_dir.name, marketplace_name=marketplace_dir.name
                )
                if self.is_installed(pid):
                    out.append(pid)
        return out


def _valid_version_segment(version: str) -> bool:
    try:
        _validate_version_segment(version)
        return True
    except ValueError:
        return False


def _validate_version_segment(version: str) -> None:
    if not version or version in {".", ".."}:
        raise ValueError("invalid plugin version")
    if not re.fullmatch(r"[A-Za-z0-9._+\-]+", version):
        raise ValueError("invalid plugin version characters")


def _version_sort_key(version: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"[.\-+]", version))


def load_plugin_state() -> dict:
    from pulsing.forge.discovery.codex_paths import forge_plugin_state_path

    path = forge_plugin_state_path()
    if not path.is_file():
        return {"enabled_plugins": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"enabled_plugins": []}
    return raw if isinstance(raw, dict) else {"enabled_plugins": []}


def save_plugin_state(state: dict) -> None:
    from pulsing.forge.discovery.codex_paths import forge_plugin_state_path

    path = forge_plugin_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def enable_plugin(plugin_id: PluginId) -> None:
    state = load_plugin_state()
    enabled = set(state.get("enabled_plugins") or [])
    enabled.add(plugin_id.id)
    state["enabled_plugins"] = sorted(enabled)
    save_plugin_state(state)
