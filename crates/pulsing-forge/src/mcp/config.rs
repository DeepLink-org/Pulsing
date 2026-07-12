//! MCP server configuration types (aligned with codex-config `mcp_types.rs`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::de::Error as SerdeError;
use serde::{Deserialize, Deserializer, Serialize};

pub const DEFAULT_MCP_SERVER_ENVIRONMENT_ID: &str = "local";

#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppToolApproval {
    #[default]
    Auto,
    Prompt,
    Approve,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpServerEnvVar {
    Name(String),
    Config {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
    },
}

impl McpServerEnvVar {
    pub fn name(&self) -> &str {
        match self {
            Self::Name(name) => name,
            Self::Config { name, .. } => name,
        }
    }

    pub fn validate_source(&self) -> Result<(), String> {
        match self {
            Self::Name(_) => Ok(()),
            Self::Config { source, .. } => match source.as_deref() {
                None | Some("local") | Some("remote") => Ok(()),
                Some(s) => Err(format!(
                    "unsupported env_vars source `{s}`; expected `local` or `remote`"
                )),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerOAuthConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum McpServerTransportConfig {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        env: Option<HashMap<String, String>>,
        #[serde(default)]
        env_vars: Vec<McpServerEnvVar>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
    },
    StreamableHttp {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bearer_token_env_var: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        http_headers: Option<HashMap<String, String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        env_http_headers: Option<HashMap<String, String>>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpServerToolConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_mode: Option<AppToolApproval>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct McpServerConfig {
    #[serde(flatten)]
    pub transport: McpServerTransportConfig,

    #[serde(default = "default_environment_id")]
    pub environment_id: String,

    #[serde(default = "default_enabled")]
    pub enabled: bool,

    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub required: bool,

    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub supports_parallel_tool_calls: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_timeout_sec: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_timeout_sec: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_tools_approval_mode: Option<AppToolApproval>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled_tools: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled_tools: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scopes: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth: Option<McpServerOAuthConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth_resource: Option<String>,

    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tools: HashMap<String, McpServerToolConfig>,
}

impl McpServerConfig {
    pub fn startup_timeout(&self) -> Duration {
        Duration::from_secs(self.startup_timeout_sec.unwrap_or(30))
    }

    pub fn tool_timeout(&self) -> Duration {
        Duration::from_secs(self.tool_timeout_sec.unwrap_or(120))
    }

    pub fn oauth_client_id(&self) -> Option<&str> {
        self.oauth.as_ref().and_then(|o| o.client_id.as_deref())
    }
}

fn default_enabled() -> bool {
    true
}

fn default_environment_id() -> String {
    DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string()
}

#[derive(Deserialize)]
struct RawMcpServerConfig {
    command: Option<String>,
    #[serde(default)]
    args: Option<Vec<String>>,
    #[serde(default)]
    env: Option<HashMap<String, String>>,
    #[serde(default)]
    env_vars: Option<Vec<McpServerEnvVar>>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    url: Option<String>,
    #[serde(default)]
    bearer_token_env_var: Option<String>,
    #[serde(default)]
    http_headers: Option<HashMap<String, String>>,
    #[serde(default)]
    env_http_headers: Option<HashMap<String, String>>,
    #[serde(default)]
    environment_id: Option<String>,
    #[serde(default)]
    startup_timeout_sec: Option<f64>,
    #[serde(default)]
    startup_timeout_ms: Option<u64>,
    #[serde(default)]
    tool_timeout_sec: Option<f64>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    required: Option<bool>,
    #[serde(default)]
    supports_parallel_tool_calls: Option<bool>,
    #[serde(default)]
    default_tools_approval_mode: Option<AppToolApproval>,
    #[serde(default)]
    enabled_tools: Option<Vec<String>>,
    #[serde(default)]
    disabled_tools: Option<Vec<String>>,
    #[serde(default)]
    scopes: Option<Vec<String>>,
    #[serde(default)]
    oauth: Option<McpServerOAuthConfig>,
    #[serde(default)]
    oauth_resource: Option<String>,
    #[serde(default)]
    tools: Option<HashMap<String, McpServerToolConfig>>,
}

impl TryFrom<RawMcpServerConfig> for McpServerConfig {
    type Error = String;

    fn try_from(raw: RawMcpServerConfig) -> Result<Self, Self::Error> {
        let startup_timeout_sec = match (raw.startup_timeout_sec, raw.startup_timeout_ms) {
            (Some(sec), _) if sec >= 0.0 => Some(sec.round() as u64),
            (None, Some(ms)) => Some(ms / 1000),
            _ => None,
        };
        let tool_timeout_sec = raw
            .tool_timeout_sec
            .filter(|s| *s >= 0.0)
            .map(|s| s.round() as u64);

        let transport = if let Some(command) = raw.command {
            let env_vars = raw.env_vars.unwrap_or_default();
            for ev in &env_vars {
                ev.validate_source()?;
            }
            McpServerTransportConfig::Stdio {
                command,
                args: raw.args.unwrap_or_default(),
                env: raw.env,
                env_vars,
                cwd: raw.cwd,
            }
        } else if let Some(url) = raw.url {
            McpServerTransportConfig::StreamableHttp {
                url,
                bearer_token_env_var: raw.bearer_token_env_var,
                http_headers: raw.http_headers,
                env_http_headers: raw.env_http_headers,
            }
        } else {
            return Err("invalid transport: need command (stdio) or url (streamable_http)".into());
        };

        Ok(Self {
            transport,
            environment_id: raw.environment_id.unwrap_or_else(default_environment_id),
            enabled: raw.enabled.unwrap_or(true),
            required: raw.required.unwrap_or(false),
            supports_parallel_tool_calls: raw.supports_parallel_tool_calls.unwrap_or(false),
            startup_timeout_sec,
            tool_timeout_sec,
            default_tools_approval_mode: raw.default_tools_approval_mode,
            enabled_tools: raw.enabled_tools,
            disabled_tools: raw.disabled_tools,
            scopes: raw.scopes,
            oauth: raw.oauth,
            oauth_resource: raw.oauth_resource,
            tools: raw.tools.unwrap_or_default(),
        })
    }
}

impl<'de> Deserialize<'de> for McpServerConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawMcpServerConfig::deserialize(deserializer)?
            .try_into()
            .map_err(SerdeError::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stdio_config() {
        let raw = r#"{"command":"npx","args":["-y","@modelcontextprotocol/server-everything"]}"#;
        let cfg: McpServerConfig = serde_json::from_str(raw).unwrap();
        match cfg.transport {
            McpServerTransportConfig::Stdio { command, .. } => assert_eq!(command, "npx"),
            _ => panic!("expected stdio"),
        }
    }
}
