# SPDX-License-Identifier: Apache-2.0
"""Codex marketplace.json discovery (aligned with codex-rs/core-plugins/marketplace.rs)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pulsing.forge.discovery.codex_manifest import load_codex_manifest
from pulsing.forge.discovery.codex_paths import (
    discover_marketplace_roots,
    find_marketplace_manifest,
)
from pulsing.forge.discovery.plugin_id import PluginId


class InstallPolicy(str, Enum):
    NOT_AVAILABLE = "NOT_AVAILABLE"
    AVAILABLE = "AVAILABLE"
    INSTALLED_BY_DEFAULT = "INSTALLED_BY_DEFAULT"


@dataclass
class MarketplacePluginSource:
    kind: str
    local_path: Path | None = None
    git_url: str | None = None
    git_path: str | None = None
    git_ref: str | None = None


@dataclass
class MarketplacePluginEntry:
    name: str
    source: MarketplacePluginSource
    installation: InstallPolicy = InstallPolicy.AVAILABLE
    marketplace_name: str = ""
    marketplace_root: Path | None = None

    @property
    def plugin_id(self) -> PluginId:
        return PluginId(plugin_name=self.name, marketplace_name=self.marketplace_name)


@dataclass
class Marketplace:
    name: str
    root: Path
    manifest_path: Path
    plugins: list[MarketplacePluginEntry] = field(default_factory=list)


def list_marketplaces(extra_roots: list[Path] | None = None) -> list[Marketplace]:
    out: list[Marketplace] = []
    for root in discover_marketplace_roots(extra_roots):
        manifest_path = find_marketplace_manifest(root)
        if manifest_path is None:
            continue
        try:
            out.append(load_marketplace(manifest_path))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return out


def load_marketplace(manifest_path: Path) -> Marketplace:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path}: marketplace must be a JSON object")
    name = str(raw.get("name") or manifest_path.parent.name)
    marketplace_root = (
        manifest_path.parent
        if manifest_path.name == "marketplace.json"
        else _marketplace_root_from_manifest(manifest_path)
    )
    plugins: list[MarketplacePluginEntry] = []
    for item in raw.get("plugins") or []:
        if not isinstance(item, dict):
            continue
        plugin_name = str(item.get("name", "")).strip()
        if not plugin_name:
            continue
        policy_raw = (item.get("policy") or {}).get("installation", "AVAILABLE")
        try:
            installation = InstallPolicy(str(policy_raw))
        except ValueError:
            installation = InstallPolicy.AVAILABLE
        source = _parse_source(item.get("source") or {}, marketplace_root)
        plugins.append(
            MarketplacePluginEntry(
                name=plugin_name,
                source=source,
                installation=installation,
                marketplace_name=name,
                marketplace_root=marketplace_root,
            )
        )
    return Marketplace(
        name=name,
        root=marketplace_root,
        manifest_path=manifest_path,
        plugins=plugins,
    )


def find_installable_plugin(plugin_id: str) -> MarketplacePluginEntry:
    pid = PluginId.parse(plugin_id)
    for marketplace in list_marketplaces():
        if marketplace.name != pid.marketplace_name:
            continue
        for plugin in marketplace.plugins:
            if plugin.name != pid.plugin_name:
                continue
            if plugin.installation == InstallPolicy.NOT_AVAILABLE:
                raise ValueError(
                    f"plugin {plugin_id!r} is not available for install in marketplace {marketplace.name!r}"
                )
            return plugin
    raise ValueError(f"plugin {plugin_id!r} was not found in any marketplace")


def materialize_plugin_source(entry: MarketplacePluginEntry) -> Path:
    source = entry.source
    if source.kind == "local":
        if source.local_path is None or not source.local_path.is_dir():
            raise ValueError(f"local plugin source missing for {entry.plugin_id.id}")
        return source.local_path.resolve()
    if source.kind == "git":
        if not source.git_url:
            raise ValueError(f"git plugin source missing url for {entry.plugin_id.id}")
        tmp = Path(tempfile.mkdtemp(prefix="forge-plugin-src-"))
        cmd = ["git", "clone", "--depth", "1"]
        if source.git_ref:
            cmd.extend(["--branch", source.git_ref])
        cmd.extend([source.git_url, str(tmp)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if source.git_path:
            sub = (tmp / source.git_path).resolve()
            if not sub.is_dir():
                raise ValueError(f"git plugin subpath missing: {source.git_path}")
            return sub
        return tmp
    raise ValueError(f"unsupported plugin source kind {source.kind!r}")


def _marketplace_root_from_manifest(manifest_path: Path) -> Path:
    for rel in (".agents/plugins/marketplace.json", ".claude-plugin/marketplace.json"):
        parts = Path(rel).parts
        current = manifest_path
        for part in reversed(parts):
            if current.name != part:
                break
            current = current.parent
        else:
            return current
    return manifest_path.parent


def _parse_source(
    raw: dict[str, Any], marketplace_root: Path
) -> MarketplacePluginSource:
    kind = str(raw.get("source", "local")).lower()
    if kind == "git":
        return MarketplacePluginSource(
            kind="git",
            git_url=str(raw.get("url", "")),
            git_path=str(raw["path"]) if raw.get("path") else None,
            git_ref=str(raw.get("ref") or raw.get("sha") or "") or None,
        )
    rel = str(raw.get("path", "."))
    local = (marketplace_root / rel).resolve()
    return MarketplacePluginSource(kind="local", local_path=local)


def copy_tree_atomic(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
