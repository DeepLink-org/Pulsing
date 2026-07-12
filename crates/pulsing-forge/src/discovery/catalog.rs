//! In-memory deferred tool catalog.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::bm25;

pub const TOOL_SEARCH_DEFAULT_LIMIT: usize = 8;
/// Upper bound on `tool_search` `limit`; guards against unbounded results for a
/// pathological/huge value from the model. Non-positive or unparsable values fall
/// back to [`TOOL_SEARCH_DEFAULT_LIMIT`] instead.
pub const TOOL_SEARCH_MAX_LIMIT: usize = 100;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeferredToolEntry {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub description: String,
    pub parameters: Value,
    pub search_text: String,
    #[serde(default)]
    pub defer_loading: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl DeferredToolEntry {
    pub fn from_function(name: &str, description: &str, parameters: Value) -> Self {
        let search_text = format!("{name} {} {}", name.replace('_', " "), description);
        Self {
            name: name.to_string(),
            namespace: None,
            description: description.to_string(),
            parameters,
            search_text,
            defer_loading: true,
            plugin_id: None,
            source: None,
        }
    }
}

use super::codex_manifest::load_codex_manifest;
use super::codex_paths::{
    TOOL_SUGGEST_PLUGIN_ALLOWLIST, discover_all_plugins_enabled, forge_plugin_state_path,
    plugins_cache_root,
};
use super::marketplace::{InstallPolicy, list_marketplaces};
use super::plugins::{
    DiscoverablePlugin, PluginManifest, load_plugin_manifests, scan_codex_plugin_dirs,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
pub struct ToolCatalog {
    deferred: Vec<DeferredToolEntry>,
    discoverable: Vec<DiscoverablePlugin>,
    installed_plugin_ids: std::collections::HashSet<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolCatalogSnapshot {
    pub tools: Vec<DeferredToolEntry>,
}

impl ToolCatalog {
    pub fn register_deferred(&mut self, entry: DeferredToolEntry) {
        if self.deferred.iter().any(|e| e.name == entry.name) {
            self.deferred.retain(|e| e.name != entry.name);
        }
        self.deferred.push(entry);
    }

    pub fn mark_plugin_installed(&mut self, plugin_id: &str) {
        self.installed_plugin_ids.insert(plugin_id.to_string());
    }

    pub fn is_plugin_installed(&self, plugin_id: &str) -> bool {
        self.installed_plugin_ids.contains(plugin_id)
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<DeferredToolEntry> {
        let docs: Vec<String> = self
            .deferred
            .iter()
            .map(|e| e.search_text.clone())
            .collect();
        let scores = bm25::bm25_scores(query, &docs);
        let mut ranked: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
            .into_iter()
            .filter(|(_, s)| *s > 0.0)
            .take(limit)
            .map(|(i, _)| self.deferred[i].clone())
            .collect()
    }

    pub fn snapshot(&self) -> ToolCatalogSnapshot {
        ToolCatalogSnapshot {
            tools: self.deferred.clone(),
        }
    }

    pub fn load_codex_plugins(&mut self, extra_dirs: &[PathBuf]) {
        let _ = self.refresh_from_codex(extra_dirs);
    }

    /// Rescan marketplaces + installed-plugin cache (Python `ToolCatalog.refresh_from_codex`).
    pub fn refresh_from_codex(&mut self, extra_dirs: &[PathBuf]) -> Result<(), String> {
        let configured = load_configured_plugin_ids();
        let installed_ids = list_installed_plugin_ids()
            .map_err(|e| format!("failed to refresh plugin catalog: {e}"))?;

        self.installed_plugin_ids = installed_ids.iter().cloned().collect();
        self.discoverable = collect_discoverable_plugins(&installed_ids, &configured, extra_dirs);

        let mut seen = std::collections::HashSet::new();
        self.deferred.clear();
        for plugin_id in installed_ids {
            match deferred_tools_for_installed_plugin(&plugin_id) {
                Ok(entries) => {
                    for entry in entries {
                        if seen.insert(entry.name.clone()) {
                            self.register_deferred(entry);
                        }
                    }
                }
                Err(_) => continue,
            }
        }
        Ok(())
    }

    pub fn list_installable(&self) -> Vec<DiscoverablePlugin> {
        self.discoverable
            .iter()
            .filter(|p| !p.installed)
            .cloned()
            .collect()
    }

    pub fn find_plugin(&self, plugin_id: &str) -> Option<DiscoverablePlugin> {
        self.discoverable
            .iter()
            .find(|p| p.id == plugin_id)
            .cloned()
    }

    pub fn list_installable_entries(&self) -> Vec<DiscoverablePlugin> {
        self.list_installable()
    }

    pub fn install_plugin(&mut self, plugin_id: &str) -> Result<Vec<DeferredToolEntry>, String> {
        let manifest_path = self
            .discoverable
            .iter()
            .find(|p| p.id == plugin_id)
            .map(|p| p.manifest_path.clone())
            .ok_or_else(|| format!("unknown plugin {plugin_id}"))?;
        let raw = std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
        let manifest: PluginManifest =
            serde_json::from_str(&raw).map_err(|e| format!("{manifest_path:?}: {e}"))?;
        self.installed_plugin_ids.insert(plugin_id.to_string());
        for plugin in &mut self.discoverable {
            if plugin.id == plugin_id {
                plugin.installed = true;
            }
        }
        let entries = manifest.deferred_tools();
        for entry in &entries {
            self.register_deferred(entry.clone());
        }
        Ok(entries)
    }
}

fn load_configured_plugin_ids() -> std::collections::HashSet<String> {
    let path = forge_plugin_state_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return std::collections::HashSet::new();
    };
    let Ok(raw) = serde_json::from_str::<serde_json::Value>(&text) else {
        return std::collections::HashSet::new();
    };
    raw.get("enabled_plugins")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn list_installed_plugin_ids() -> Result<Vec<String>, std::io::Error> {
    let cache = plugins_cache_root();
    let mut out = Vec::new();
    if !cache.is_dir() {
        return Ok(out);
    }
    for marketplace_dir in std::fs::read_dir(&cache)? {
        let marketplace_dir = marketplace_dir?;
        if !marketplace_dir.file_type()?.is_dir() {
            continue;
        }
        let marketplace_name = marketplace_dir.file_name().to_string_lossy().to_string();
        for plugin_dir in std::fs::read_dir(marketplace_dir.path())? {
            let plugin_dir = plugin_dir?;
            if !plugin_dir.file_type()?.is_dir() {
                continue;
            }
            let plugin_name = plugin_dir.file_name().to_string_lossy().to_string();
            let plugin_id = format!("{plugin_name}@{marketplace_name}");
            if is_plugin_installed(&plugin_id) {
                out.push(plugin_id);
            }
        }
    }
    Ok(out)
}

fn is_plugin_installed(plugin_id: &str) -> bool {
    let Some((plugin_name, marketplace_name)) = plugin_id.rsplit_once('@') else {
        return false;
    };
    let base = plugins_cache_root()
        .join(marketplace_name)
        .join(plugin_name);
    if !base.is_dir() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(&base) else {
        return false;
    };
    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        if super::codex_paths::find_plugin_manifest_path(&entry.path()).is_some() {
            return true;
        }
    }
    false
}

fn collect_discoverable_plugins(
    installed_ids: &[String],
    configured: &std::collections::HashSet<String>,
    extra_dirs: &[PathBuf],
) -> Vec<DiscoverablePlugin> {
    let discover_all = discover_all_plugins_enabled();
    let installed: std::collections::HashSet<&str> =
        installed_ids.iter().map(String::as_str).collect();
    let mut out = Vec::new();

    for marketplace in list_marketplaces(extra_dirs) {
        for entry in &marketplace.plugins {
            let plugin_id = entry.plugin_id();
            if installed.contains(plugin_id.as_str()) {
                continue;
            }
            if entry.installation == InstallPolicy::NotAvailable {
                continue;
            }
            let in_allowlist = TOOL_SUGGEST_PLUGIN_ALLOWLIST.contains(&plugin_id.as_str());
            let is_configured = configured.contains(&plugin_id);
            if !discover_all && !in_allowlist && !is_configured {
                continue;
            }
            let manifest = entry
                .source
                .local_path
                .as_ref()
                .filter(|p| p.is_dir())
                .and_then(|p| load_codex_manifest(p).ok());
            let name = manifest
                .as_ref()
                .and_then(|m| m.display_name.clone())
                .or_else(|| manifest.as_ref().map(|m| m.name.clone()))
                .unwrap_or_else(|| entry.name.clone());
            let description = manifest.as_ref().and_then(|m| m.description.clone());
            let manifest_path = manifest
                .as_ref()
                .map(|m| m.manifest_path.clone())
                .unwrap_or_else(|| entry.marketplace_root.clone());
            out.push(DiscoverablePlugin {
                id: plugin_id,
                name,
                description,
                remote_plugin_id: remote_plugin_id(&entry.marketplace_name, &entry.name),
                has_skills: manifest.as_ref().is_some_and(|m| m.has_skills),
                mcp_server_names: manifest
                    .as_ref()
                    .map(|m| m.mcp_server_names.clone())
                    .unwrap_or_default(),
                app_connector_ids: manifest
                    .as_ref()
                    .map(|m| m.app_connector_ids.clone())
                    .unwrap_or_default(),
                manifest_path,
                installed: false,
            });
        }
    }

    // Legacy flat plugin dirs (tests + FORGE_PLUGIN_DIRS).
    let dirs = scan_codex_plugin_dirs(extra_dirs);
    for (manifest, path) in load_plugin_manifests(&dirs) {
        if installed.contains(manifest.id.as_str()) {
            continue;
        }
        out.retain(|p| p.id != manifest.id);
        out.push(manifest.to_discoverable(&path, false));
    }
    out
}

fn remote_plugin_id(marketplace_name: &str, plugin_name: &str) -> Option<String> {
    if marketplace_name.ends_with("-remote") {
        Some(format!("plugins~Plugin_{plugin_name}"))
    } else {
        None
    }
}

fn deferred_tools_for_installed_plugin(plugin_id: &str) -> Result<Vec<DeferredToolEntry>, String> {
    let Some((plugin_name, marketplace_name)) = plugin_id.rsplit_once('@') else {
        return Ok(Vec::new());
    };
    let base = plugins_cache_root()
        .join(marketplace_name)
        .join(plugin_name);
    let root =
        active_plugin_root(&base).ok_or_else(|| format!("plugin not installed: {plugin_id}"))?;
    let manifest = load_codex_manifest(&root)?;
    Ok(manifest
        .mcp_server_names
        .iter()
        .map(|server| {
            let ns = format!("mcp__{server}");
            let mut entry = DeferredToolEntry::from_function(
                &ns,
                &format!(
                    "MCP server {server} from plugin {}",
                    manifest.display_name.as_deref().unwrap_or(&manifest.name)
                ),
                serde_json::json!({"type": "object", "properties": {}}),
            );
            entry.defer_loading = true;
            entry.namespace = Some(ns.clone());
            entry.plugin_id = Some(plugin_id.to_string());
            entry.source = Some("codex_mcp_server".into());
            entry
        })
        .collect())
}

fn active_plugin_root(base: &Path) -> Option<PathBuf> {
    if !base.is_dir() {
        return None;
    }
    let Ok(entries) = std::fs::read_dir(base) else {
        return None;
    };
    let mut versions: Vec<PathBuf> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    if versions.is_empty() {
        return None;
    }
    if let Some(local) = versions
        .iter()
        .find(|p| p.file_name().and_then(|s| s.to_str()) == Some("local"))
    {
        return Some(local.clone());
    }
    versions.sort_by(|a, b| {
        let av = a.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let bv = b.file_name().and_then(|s| s.to_str()).unwrap_or("");
        av.cmp(bv)
    });
    versions.last().cloned()
}
