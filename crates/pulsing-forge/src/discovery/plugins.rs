//! Codex-compatible plugin manifest loading.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::catalog::DeferredToolEntry;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginToolDef {
    #[serde(default)]
    pub r#type: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Value,
    #[serde(default)]
    pub defer_loading: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub remote_plugin_id: Option<String>,
    #[serde(default)]
    pub has_skills: bool,
    #[serde(default)]
    pub mcp_server_names: Vec<String>,
    #[serde(default)]
    pub app_connector_ids: Vec<String>,
    #[serde(default)]
    pub tools: Vec<PluginToolDef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoverablePlugin {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub remote_plugin_id: Option<String>,
    pub has_skills: bool,
    pub mcp_server_names: Vec<String>,
    pub app_connector_ids: Vec<String>,
    pub manifest_path: PathBuf,
    pub installed: bool,
}

pub fn scan_codex_plugin_dirs(extra_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(home) = dirs_home() {
        out.push(home.join(".codex").join("plugins"));
    }
    for d in extra_dirs {
        out.push(d.clone());
    }
    if let Ok(env) = std::env::var("FORGE_PLUGIN_DIRS") {
        for part in env.split(':').filter(|s| !s.is_empty()) {
            out.push(PathBuf::from(part));
        }
    }
    out
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

pub fn load_plugin_manifests(dirs: &[PathBuf]) -> Vec<(PluginManifest, PathBuf)> {
    let mut out = Vec::new();
    for root in dirs {
        if !root.is_dir() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                for name in ["plugin.json", "manifest.json", "codex-plugin.json"] {
                    let mf = path.join(name);
                    if mf.is_file()
                        && let Ok(m) = read_manifest(&mf)
                    {
                        out.push((m, mf));
                    }
                }
            } else if path.extension().is_some_and(|e| e == "json")
                && let Ok(m) = read_manifest(&path)
            {
                out.push((m, path));
            }
        }
    }
    out
}

fn read_manifest(path: &Path) -> Result<PluginManifest, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))
}

impl PluginManifest {
    pub fn to_discoverable(&self, manifest_path: &Path, installed: bool) -> DiscoverablePlugin {
        DiscoverablePlugin {
            id: self.id.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            remote_plugin_id: self.remote_plugin_id.clone(),
            has_skills: self.has_skills,
            mcp_server_names: self.mcp_server_names.clone(),
            app_connector_ids: self.app_connector_ids.clone(),
            manifest_path: manifest_path.to_path_buf(),
            installed,
        }
    }

    pub fn deferred_tools(&self) -> Vec<DeferredToolEntry> {
        self.tools
            .iter()
            .map(|t| {
                let mut entry =
                    DeferredToolEntry::from_function(&t.name, &t.description, t.parameters.clone());
                entry.defer_loading = t.defer_loading;
                entry.plugin_id = Some(self.id.clone());
                entry.source = Some("codex_plugin".into());
                entry
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_manifest() {
        let raw = r#"{
            "id": "demo",
            "name": "Demo Plugin",
            "tools": [{"name": "demo_tool", "description": "hello", "parameters": {"type":"object"}}]
        }"#;
        let m: PluginManifest = serde_json::from_str(raw).unwrap();
        assert_eq!(m.tools.len(), 1);
        assert_eq!(m.deferred_tools()[0].name, "demo_tool");
    }
}
