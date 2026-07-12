//! Codex marketplace.json discovery (aligned with python/pulsing/forge/discovery/marketplace.py).

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::codex_paths::{discover_marketplace_roots, find_marketplace_manifest};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallPolicy {
    NotAvailable,
    Available,
    InstalledByDefault,
}

#[derive(Clone, Debug)]
pub struct MarketplacePluginSource {
    pub kind: String,
    pub local_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct MarketplacePluginEntry {
    pub name: String,
    pub source: MarketplacePluginSource,
    pub installation: InstallPolicy,
    pub marketplace_name: String,
    pub marketplace_root: PathBuf,
}

impl MarketplacePluginEntry {
    pub fn plugin_id(&self) -> String {
        format!("{}@{}", self.name, self.marketplace_name)
    }
}

#[derive(Clone, Debug)]
pub struct Marketplace {
    pub name: String,
    pub root: PathBuf,
    pub manifest_path: PathBuf,
    pub plugins: Vec<MarketplacePluginEntry>,
}

pub fn list_marketplaces(extra_roots: &[PathBuf]) -> Vec<Marketplace> {
    let mut out = Vec::new();
    for root in discover_marketplace_roots(extra_roots) {
        let Some(manifest_path) = find_marketplace_manifest(&root) else {
            continue;
        };
        if let Ok(marketplace) = load_marketplace(&manifest_path) {
            out.push(marketplace);
        }
    }
    out
}

pub fn load_marketplace(manifest_path: &Path) -> Result<Marketplace, String> {
    let raw = std::fs::read_to_string(manifest_path).map_err(|e| e.to_string())?;
    let value: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let obj = value.as_object().ok_or_else(|| {
        format!(
            "{}: marketplace must be a JSON object",
            manifest_path.display()
        )
    })?;
    let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or_else(|| {
        manifest_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("marketplace")
    });
    let marketplace_root =
        if manifest_path.file_name().and_then(|s| s.to_str()) == Some("marketplace.json") {
            manifest_path
                .parent()
                .unwrap_or(manifest_path)
                .to_path_buf()
        } else {
            marketplace_root_from_manifest(manifest_path)
        };
    let mut plugins = Vec::new();
    if let Some(items) = obj.get("plugins").and_then(|v| v.as_array()) {
        for item in items {
            let Some(entry_obj) = item.as_object() else {
                continue;
            };
            let plugin_name = entry_obj
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if plugin_name.is_empty() {
                continue;
            }
            let policy_raw = entry_obj
                .get("policy")
                .and_then(|v| v.as_object())
                .and_then(|p| p.get("installation"))
                .and_then(|v| v.as_str())
                .unwrap_or("AVAILABLE");
            let installation = match policy_raw {
                "NOT_AVAILABLE" => InstallPolicy::NotAvailable,
                "INSTALLED_BY_DEFAULT" => InstallPolicy::InstalledByDefault,
                _ => InstallPolicy::Available,
            };
            let source = parse_source(
                entry_obj.get("source").and_then(|v| v.as_object()),
                &marketplace_root,
            );
            plugins.push(MarketplacePluginEntry {
                name: plugin_name,
                source,
                installation,
                marketplace_name: name.to_string(),
                marketplace_root: marketplace_root.clone(),
            });
        }
    }
    Ok(Marketplace {
        name: name.to_string(),
        root: marketplace_root.clone(),
        manifest_path: manifest_path.to_path_buf(),
        plugins,
    })
}

fn marketplace_root_from_manifest(manifest_path: &Path) -> PathBuf {
    for rel in super::codex_paths::MARKETPLACE_MANIFEST_RELATIVE_PATHS {
        let parts: Vec<_> = Path::new(rel).components().collect();
        let mut current = manifest_path.to_path_buf();
        let mut matched = true;
        for part in parts.iter().rev() {
            if current.file_name() != part.as_os_str().into() {
                matched = false;
                break;
            }
            current = current.parent().unwrap_or(&current).to_path_buf();
        }
        if matched {
            return current;
        }
    }
    manifest_path
        .parent()
        .unwrap_or(manifest_path)
        .to_path_buf()
}

fn parse_source(
    raw: Option<&serde_json::Map<String, Value>>,
    marketplace_root: &Path,
) -> MarketplacePluginSource {
    let Some(raw) = raw else {
        return MarketplacePluginSource {
            kind: "local".into(),
            local_path: Some(marketplace_root.to_path_buf()),
        };
    };
    let kind = raw
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("local")
        .to_ascii_lowercase();
    if kind == "git" {
        return MarketplacePluginSource {
            kind: "git".into(),
            local_path: None,
        };
    }
    let rel = raw.get("path").and_then(|v| v.as_str()).unwrap_or(".");
    let local = marketplace_root.join(rel);
    MarketplacePluginSource {
        kind: "local".into(),
        local_path: Some(local),
    }
}
