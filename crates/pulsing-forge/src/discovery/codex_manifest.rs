//! Parse Codex `.codex-plugin/plugin.json` manifests.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::codex_paths::find_plugin_manifest_path;

#[derive(Clone, Debug)]
pub struct CodexPluginManifest {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub display_name: Option<String>,
    pub has_skills: bool,
    pub mcp_server_names: Vec<String>,
    pub app_connector_ids: Vec<String>,
    pub manifest_path: PathBuf,
}

pub fn load_codex_manifest(plugin_root: &Path) -> Result<CodexPluginManifest, String> {
    let manifest_path = find_plugin_manifest_path(plugin_root)
        .ok_or_else(|| format!("missing plugin manifest under {}", plugin_root.display()))?;
    let raw = std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
    let value: Value =
        serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let obj = value.as_object().ok_or_else(|| {
        format!(
            "{}: manifest must be a JSON object",
            manifest_path.display()
        )
    })?;
    parse_codex_manifest(obj, plugin_root, manifest_path)
}

fn parse_codex_manifest(
    raw: &serde_json::Map<String, Value>,
    plugin_root: &Path,
    manifest_path: PathBuf,
) -> Result<CodexPluginManifest, String> {
    let interface = raw
        .get("interface")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let display = interface
        .get("displayName")
        .or_else(|| interface.get("display_name"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let desc = raw
        .get("description")
        .or_else(|| interface.get("shortDescription"))
        .or_else(|| interface.get("short_description"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let mcp_ref = raw
        .get("mcpServers")
        .or_else(|| raw.get("mcp_servers"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let apps_ref = raw.get("apps").and_then(|v| v.as_str()).map(str::to_string);
    let skills = raw.get("skills").and_then(|v| v.as_str()).is_some();
    let name = raw.get("name").and_then(|v| v.as_str()).unwrap_or_else(|| {
        plugin_root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("plugin")
    });
    let version = raw
        .get("version")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Ok(CodexPluginManifest {
        name: name.to_string(),
        version,
        description: desc,
        display_name: display,
        has_skills: skills,
        mcp_server_names: extract_mcp_server_names(plugin_root, mcp_ref.as_deref()),
        app_connector_ids: extract_app_connector_ids(plugin_root, apps_ref.as_deref()),
        manifest_path,
    })
}

fn resolve_relative(plugin_root: &Path, reference: Option<&str>) -> Option<PathBuf> {
    let reference = reference?;
    let path = plugin_root.join(reference);
    if path.exists() { Some(path) } else { None }
}

fn extract_mcp_server_names(plugin_root: &Path, mcp_ref: Option<&str>) -> Vec<String> {
    let Some(path) = resolve_relative(plugin_root, mcp_ref) else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(raw) = serde_json::from_str::<Value>(&text) else {
        return Vec::new();
    };
    let servers = raw.get("mcpServers").or(Some(&raw));
    if let Some(obj) = servers.and_then(|v| v.as_object()) {
        let mut names: Vec<String> = obj
            .keys()
            .filter(|k| !k.trim().is_empty())
            .cloned()
            .collect();
        names.sort();
        return names;
    }
    Vec::new()
}

fn extract_app_connector_ids(plugin_root: &Path, apps_ref: Option<&str>) -> Vec<String> {
    let Some(path) = resolve_relative(plugin_root, apps_ref) else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(raw) = serde_json::from_str::<Value>(&text) else {
        return Vec::new();
    };
    let Some(obj) = raw.as_object() else {
        return Vec::new();
    };
    let connectors = obj
        .get("connectors")
        .or_else(|| obj.get("apps"))
        .unwrap_or(&raw);
    if let Some(list) = connectors.as_array() {
        return list
            .iter()
            .filter_map(|c| {
                c.as_object()
                    .and_then(|o| o.get("id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect();
    }
    if let Some(map) = connectors.as_object() {
        let mut ids: Vec<String> = map.keys().cloned().collect();
        ids.sort();
        return ids;
    }
    Vec::new()
}
