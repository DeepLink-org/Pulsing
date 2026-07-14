//! Codex home paths and config.toml MCP server loading.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::config::McpServerConfig;

pub fn codex_home() -> PathBuf {
    if let Ok(raw) = std::env::var("CODEX_HOME") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    dirs_home().join(".codex")
}

pub fn credentials_path() -> PathBuf {
    codex_home().join(".credentials.json")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

pub fn config_toml_path() -> PathBuf {
    codex_home().join("config.toml")
}

/// Load `[mcp_servers.<name>]` from `~/.codex/config.toml`.
pub fn load_config_mcp_servers() -> HashMap<String, McpServerConfig> {
    let path = config_toml_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return HashMap::new();
    };
    parse_config_toml_mcp_servers(&text)
}

pub fn parse_config_toml_mcp_servers(text: &str) -> HashMap<String, McpServerConfig> {
    let Ok(doc) = text.parse::<toml::Table>() else {
        return HashMap::new();
    };
    let Some(mcp_servers) = doc.get("mcp_servers").and_then(|v| v.as_table()) else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    for (name, value) in mcp_servers {
        let Some(table) = value.as_table() else {
            continue;
        };
        let table_value = toml_table_to_json(table);
        match serde_json::from_value::<McpServerConfig>(table_value) {
            Ok(cfg) => {
                out.insert(name.clone(), cfg);
            }
            Err(err) => {
                tracing::warn!(server = %name, error = %err, "skip invalid mcp_servers entry");
            }
        }
    }
    out
}

fn toml_table_to_json(table: &toml::Table) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in table {
        map.insert(k.clone(), toml_value_to_json(v));
    }
    serde_json::Value::Object(map)
}

fn toml_value_to_json(v: &toml::Value) -> serde_json::Value {
    match v {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Integer(i) => serde_json::Value::Number((*i).into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
        toml::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(toml_value_to_json).collect())
        }
        toml::Value::Table(t) => toml_table_to_json(t),
    }
}

pub fn plugins_cache_root() -> PathBuf {
    codex_home().join("plugins/cache")
}

pub fn resolve_plugin_root(marketplace: &str, name: &str, version: &str) -> PathBuf {
    plugins_cache_root()
        .join(marketplace)
        .join(name)
        .join(version)
}

pub fn find_plugin_mcp_json(plugin_root: &Path, relative: &str) -> Option<PathBuf> {
    let path = plugin_root.join(relative);
    if path.is_file() { Some(path) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mcp_servers_toml() {
        let text = r#"
[mcp_servers.demo]
command = "echo"
args = ["hello"]
enabled = true
"#;
        let servers = parse_config_toml_mcp_servers(text);
        assert_eq!(servers.len(), 1);
        assert!(servers.contains_key("demo"));
    }
}
