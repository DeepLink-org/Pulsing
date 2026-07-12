//! Parse plugin `.mcp.json` files (aligned with codex-mcp `plugin_config.rs`).

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map as JsonMap, Value as JsonValue};
use tracing::warn;

use super::config::McpServerConfig;

#[derive(Clone, Copy, Debug)]
pub enum PluginMcpServerPlacement<'a> {
    Declared,
    Environment { environment_id: &'a str },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginMcpServerParseError {
    pub name: String,
    pub message: String,
}

#[derive(Debug, Default, PartialEq)]
pub struct PluginMcpConfigParseOutcome {
    pub servers: BTreeMap<String, McpServerConfig>,
    pub errors: Vec<PluginMcpServerParseError>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PluginMcpServersFile {
    mcp_servers: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PluginMcpFile {
    McpServersObject(PluginMcpServersFile),
    ServerMap(BTreeMap<String, JsonValue>),
}

impl PluginMcpFile {
    fn into_mcp_servers(self) -> BTreeMap<String, JsonValue> {
        match self {
            Self::McpServersObject(file) => file.mcp_servers,
            Self::ServerMap(mcp_servers) => mcp_servers,
        }
    }
}

pub fn parse_plugin_mcp_config(
    plugin_root: &Path,
    contents: &str,
    placement: PluginMcpServerPlacement<'_>,
) -> Result<PluginMcpConfigParseOutcome, serde_json::Error> {
    let parsed = serde_json::from_str::<PluginMcpFile>(contents)?;
    let mut outcome = PluginMcpConfigParseOutcome::default();

    for (name, config_value) in parsed.into_mcp_servers() {
        match normalize_plugin_mcp_server(plugin_root, config_value, placement) {
            Ok(config) => {
                outcome.servers.insert(name, config);
            }
            Err(message) => outcome
                .errors
                .push(PluginMcpServerParseError { name, message }),
        }
    }

    Ok(outcome)
}

fn normalize_plugin_mcp_server(
    plugin_root: &Path,
    value: JsonValue,
    placement: PluginMcpServerPlacement<'_>,
) -> Result<McpServerConfig, String> {
    let mut object = normalize_plugin_mcp_server_value(plugin_root, value, placement);
    if let PluginMcpServerPlacement::Environment { environment_id } = placement {
        object.insert(
            "environment_id".to_string(),
            JsonValue::String(environment_id.to_string()),
        );
        if object.contains_key("command") {
            match object.remove("cwd") {
                Some(JsonValue::String(cwd)) => object.insert(
                    "cwd".to_string(),
                    JsonValue::String(
                        executor_plugin_cwd(plugin_root, &cwd)?
                            .to_string_lossy()
                            .into_owned(),
                    ),
                ),
                Some(JsonValue::Null) | None => object.insert(
                    "cwd".to_string(),
                    JsonValue::String(plugin_root.to_string_lossy().into_owned()),
                ),
                Some(value) => object.insert("cwd".to_string(), value),
            };
        }
    }

    let mut config = serde_json::from_value::<McpServerConfig>(JsonValue::Object(object))
        .map_err(|e| e.to_string())?;
    if matches!(placement, PluginMcpServerPlacement::Environment { .. }) {
        bind_environment_env_vars(&mut config)?;
    }
    Ok(config)
}

fn bind_environment_env_vars(config: &mut McpServerConfig) -> Result<(), String> {
    use super::config::{McpServerEnvVar, McpServerTransportConfig};

    let McpServerTransportConfig::Stdio { env_vars, .. } = &mut config.transport else {
        return Ok(());
    };
    let is_local = config.environment_id == super::config::DEFAULT_MCP_SERVER_ENVIRONMENT_ID;
    for env_var in env_vars {
        match env_var {
            McpServerEnvVar::Config { name, source } if source.is_none() && !is_local => {
                *source = Some("remote".to_string());
            }
            McpServerEnvVar::Config {
                name,
                source: Some(s),
                ..
            } if is_local && s == "remote" => {
                return Err(format!(
                    "env_vars entry `{name}` cannot use source `remote` in a local environment"
                ));
            }
            McpServerEnvVar::Config {
                name,
                source: Some(s),
                ..
            } if !is_local && s == "local" => {
                return Err(format!(
                    "env_vars entry `{name}` cannot use source `local` in an executor-owned plugin"
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn normalize_plugin_mcp_server_value(
    plugin_root: &Path,
    value: JsonValue,
    placement: PluginMcpServerPlacement<'_>,
) -> JsonMap<String, JsonValue> {
    let mut object = match value {
        JsonValue::Object(object) => object,
        _ => return JsonMap::new(),
    };

    if let Some(JsonValue::String(transport_type)) = object.remove("type") {
        match transport_type.as_str() {
            "http" | "streamable_http" | "streamable-http" | "stdio" => {}
            other => {
                warn!(
                    plugin = %plugin_root.display(),
                    transport = other,
                    "plugin MCP server uses an unknown transport type"
                );
            }
        }
    }

    if let Some(JsonValue::Object(mut oauth)) = object.remove("oauth") {
        if oauth.remove("callbackPort").is_some() {
            warn!(
                plugin = %plugin_root.display(),
                "plugin MCP OAuth callbackPort is ignored; Forge uses global MCP OAuth callback settings"
            );
        }
        if let Some(client_id) = oauth.remove("clientId") {
            oauth.entry("client_id".to_string()).or_insert(client_id);
        }
        if !oauth.is_empty() {
            object.insert("oauth".to_string(), JsonValue::Object(oauth));
        }
    }

    if matches!(placement, PluginMcpServerPlacement::Declared) {
        if let Some(JsonValue::String(cwd)) = object.get("cwd") {
            if !Path::new(cwd).is_absolute() {
                object.insert(
                    "cwd".to_string(),
                    JsonValue::String(plugin_root.join(cwd).display().to_string()),
                );
            }
        }
    }

    object
}

fn executor_plugin_cwd(plugin_root: &Path, cwd: &str) -> Result<PathBuf, String> {
    let path = Path::new(cwd);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(format!(
                "plugin MCP cwd must stay within plugin root: {cwd}"
            ));
        }
    }
    Ok(plugin_root.join(path))
}

pub fn parse_plugin_mcp_file(
    plugin_root: &Path,
    mcp_path: &Path,
) -> Result<PluginMcpConfigParseOutcome, String> {
    let contents = std::fs::read_to_string(mcp_path).map_err(|e| e.to_string())?;
    parse_plugin_mcp_config(plugin_root, &contents, PluginMcpServerPlacement::Declared)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mcp_servers_wrapper() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{"mcpServers":{"demo":{"command":"echo","args":["hi"]}}}"#;
        let outcome =
            parse_plugin_mcp_config(dir.path(), json, PluginMcpServerPlacement::Declared).unwrap();
        assert_eq!(outcome.servers.len(), 1);
        assert!(outcome.errors.is_empty());
    }

    #[test]
    fn parse_flat_server_map() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{"github":{"command":"npx","args":["-y","pkg"]}}"#;
        let outcome =
            parse_plugin_mcp_config(dir.path(), json, PluginMcpServerPlacement::Declared).unwrap();
        assert!(outcome.servers.contains_key("github"));
    }
}
