//! Codex home + marketplace roots (aligned with python/pulsing/forge/discovery/codex_paths.py).

use std::path::{Path, PathBuf};

pub const PLUGINS_CACHE_DIR: &str = "plugins/cache";

pub const MARKETPLACE_MANIFEST_RELATIVE_PATHS: &[&str] = &[
    ".agents/plugins/marketplace.json",
    ".claude-plugin/marketplace.json",
];

pub const PLUGIN_MANIFEST_RELATIVE_PATHS: &[&str] =
    &[".codex-plugin/plugin.json", ".claude-plugin/plugin.json"];

/// Codex tool_suggest fallback allowlist (core-plugins/src/discoverable.rs).
pub const TOOL_SUGGEST_PLUGIN_ALLOWLIST: &[&str] = &[
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
];

pub fn codex_home() -> PathBuf {
    if let Ok(raw) = std::env::var("CODEX_HOME") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    dirs_home().join(".codex")
}

pub fn plugins_cache_root() -> PathBuf {
    codex_home().join(PLUGINS_CACHE_DIR)
}

pub fn forge_plugin_state_path() -> PathBuf {
    codex_home().join("forge").join("plugin_state.json")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

pub fn discover_marketplace_roots(extra_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for base in [dirs_home(), codex_home()] {
        let agents = base.join(".agents").join("plugins");
        if agents.is_dir() {
            roots.push(agents);
        }
        for rel in MARKETPLACE_MANIFEST_RELATIVE_PATHS {
            let candidate = base.join(rel);
            if candidate.is_file() {
                roots.push(candidate.parent().unwrap_or(&base).to_path_buf());
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for rel in MARKETPLACE_MANIFEST_RELATIVE_PATHS {
            let candidate = cwd.join(rel);
            if candidate.is_file() {
                roots.push(candidate.parent().unwrap_or(&cwd).to_path_buf());
            }
        }
    }
    if let Ok(env) = std::env::var("FORGE_PLUGIN_DIRS") {
        for part in env.split(':').filter(|s| !s.is_empty()) {
            roots.push(PathBuf::from(part));
        }
    }
    roots.extend(extra_roots.iter().cloned());

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for root in roots {
        let resolved = root.canonicalize().unwrap_or(root);
        if seen.insert(resolved.clone()) {
            out.push(resolved);
        }
    }
    out
}

pub fn find_marketplace_manifest(root: &Path) -> Option<PathBuf> {
    for rel in MARKETPLACE_MANIFEST_RELATIVE_PATHS {
        let candidate = root.join(rel);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let direct = root.join("marketplace.json");
    if direct.is_file() {
        return Some(direct);
    }
    None
}

pub fn find_plugin_manifest_path(plugin_root: &Path) -> Option<PathBuf> {
    for rel in PLUGIN_MANIFEST_RELATIVE_PATHS {
        let candidate = plugin_root.join(rel);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn discover_all_plugins_enabled() -> bool {
    matches!(
        std::env::var("FORGE_PLUGIN_DISCOVER_ALL")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}
