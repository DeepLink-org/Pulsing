# SPDX-License-Identifier: Apache-2.0
"""Codex home + plugin directory layout (aligned with codex-rs/core-plugins)."""

from __future__ import annotations

import os
from pathlib import Path

PLUGINS_CACHE_DIR = "plugins/cache"
PLUGINS_DATA_DIR = "plugins/data"
DEFAULT_PLUGIN_VERSION = "local"

MARKETPLACE_MANIFEST_RELATIVE_PATHS = (
    ".agents/plugins/marketplace.json",
    ".claude-plugin/marketplace.json",
)

PLUGIN_MANIFEST_RELATIVE_PATHS = (
    ".codex-plugin/plugin.json",
    ".claude-plugin/plugin.json",
)

# Codex tool_suggest fallback allowlist (core-plugins/src/discoverable.rs).
TOOL_SUGGEST_PLUGIN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "github@openai-curated",
        "notion@openai-curated",
        "slack@openai-curated",
        "gmail@openai-curated",
        "google-calendar@openai-curated",
        "google-drive@openai-curated",
        "openai-developers@openai-curated",
        "canva@openai-curated",
        "teams@openai-curated",
        "sharepoint@openai-curated",
        "outlook-email@openai-curated",
        "outlook-calendar@openai-curated",
        "linear@openai-curated",
        "figma@openai-curated",
        "github@openai-curated-remote",
        "notion@openai-curated-remote",
        "slack@openai-curated-remote",
        "gmail@openai-curated-remote",
        "google-calendar@openai-curated-remote",
        "google-drive@openai-curated-remote",
        "openai-developers@openai-curated-remote",
        "canva@openai-curated-remote",
        "teams@openai-curated-remote",
        "sharepoint@openai-curated-remote",
        "outlook-email@openai-curated-remote",
        "outlook-calendar@openai-curated-remote",
        "linear@openai-curated-remote",
        "figma@openai-curated-remote",
        "chrome@openai-bundled",
        "computer-use@openai-bundled",
    }
)


def codex_home() -> Path:
    raw = os.environ.get("CODEX_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".codex").resolve()


def plugins_cache_root() -> Path:
    return codex_home() / PLUGINS_CACHE_DIR


def plugins_data_root() -> Path:
    return codex_home() / PLUGINS_DATA_DIR


def forge_plugin_state_path() -> Path:
    return codex_home() / "forge" / "plugin_state.json"


def scan_codex_plugin_dirs_legacy(extra_dirs: list[str] | None = None) -> list[Path]:
    """Legacy flat scan roots — prefer marketplace + cache via ToolCatalog.refresh_from_codex."""
    out: list[Path] = [plugins_cache_root(), codex_home() / "plugins"]
    for part in os.environ.get("FORGE_PLUGIN_DIRS", "").split(":"):
        if part.strip():
            out.append(Path(part))
    for d in extra_dirs or []:
        out.append(Path(d))
    return out


def discover_marketplace_roots(extra_roots: list[Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    for base in (Path.home(), codex_home()):
        agents = base / ".agents" / "plugins"
        if agents.is_dir():
            roots.append(agents)
        for rel in MARKETPLACE_MANIFEST_RELATIVE_PATHS:
            candidate = base / rel
            if candidate.is_file():
                roots.append(candidate.parent)
    cwd = Path.cwd()
    for rel in MARKETPLACE_MANIFEST_RELATIVE_PATHS:
        candidate = cwd / rel
        if candidate.is_file():
            roots.append(candidate.parent)
    for part in os.environ.get("FORGE_PLUGIN_DIRS", "").split(":"):
        if part.strip():
            roots.append(Path(part).expanduser())
    for root in extra_roots or []:
        roots.append(root)
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def find_marketplace_manifest(root: Path) -> Path | None:
    for rel in MARKETPLACE_MANIFEST_RELATIVE_PATHS:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    direct = root / "marketplace.json"
    if direct.is_file():
        return direct
    return None
