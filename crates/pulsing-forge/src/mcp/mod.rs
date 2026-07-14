//! Codex-aligned MCP client runtime + Forge integration.
//!
//! Ports essential behavior of `codex-mcp` + `codex-rmcp-client` without the full Codex workspace.

mod catalog;
mod client;
mod codex_home;
mod config;
mod connection_manager;
mod oauth;
mod plugin_config;
mod resources;
mod tools;

pub use catalog::{
    McpCatalogBuilder, McpServerConflict, McpServerConflictAction, McpServerRegistration,
    McpServerSource, ResolvedMcpCatalog, ResolvedMcpServer, build_default_catalog,
};
pub use client::{McpClientError, McpManagedClient};
pub use codex_home::{codex_home, credentials_path, load_config_mcp_servers, plugins_cache_root};
pub use config::{
    AppToolApproval, DEFAULT_MCP_SERVER_ENVIRONMENT_ID, McpServerConfig, McpServerEnvVar,
    McpServerOAuthConfig, McpServerToolConfig, McpServerTransportConfig,
};
pub use connection_manager::{McpConnectionManager, McpStartupFailure, McpStartupStatus};
pub use oauth::{OAuthCredentialsStore, OAuthLoginHandle, perform_oauth_login};
pub use plugin_config::{
    PluginMcpConfigParseOutcome, PluginMcpServerParseError, PluginMcpServerPlacement,
    parse_plugin_mcp_config,
};
pub use resources::{
    MAX_MCP_RESOURCE_BYTES, enforce_resource_size_limit, validate_mcp_resource_uri,
};
pub use tools::{
    ToolFilter, ToolInfo, tool_input_schema_json, tool_is_model_visible, tool_spec_for_model,
};

pub const DIRECT_MCP_TOOL_EXPOSURE_THRESHOLD: usize = 100;
pub const LEGACY_MCP_TOOL_NAME_PREFIX: &str = "mcp__";
pub const DEFAULT_STARTUP_TIMEOUT_SECS: u64 = 30;
pub const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 120;

use std::path::Path;
use std::sync::Arc;

pub struct McpRuntime {
    pub catalog: ResolvedMcpCatalog,
    pub manager: McpConnectionManager,
}

impl McpRuntime {
    pub async fn from_codex_home(prefix_mcp_tool_names: bool) -> Self {
        let catalog = load_codex_mcp_catalog();
        let manager = McpConnectionManager::start(&catalog, prefix_mcp_tool_names).await;
        Self { catalog, manager }
    }

    pub fn tool_model_names(&self) -> Vec<String> {
        self.manager
            .list_all_tools()
            .iter()
            .map(|t| t.model_tool_name(true))
            .collect()
    }

    pub fn tool_specs_for_model(&self) -> Vec<serde_json::Value> {
        self.manager
            .list_all_tools()
            .iter()
            .map(|t| tool_spec_for_model(t, true))
            .collect()
    }
}

pub fn load_codex_mcp_catalog() -> ResolvedMcpCatalog {
    let config_servers = load_config_mcp_servers();
    let mut plugin_servers = Vec::new();
    let cache = plugins_cache_root();
    if cache.is_dir() {
        collect_plugin_mcp_servers(&cache, &mut plugin_servers);
    }
    build_default_catalog(plugin_servers, config_servers)
}

fn collect_plugin_mcp_servers(
    cache: &Path,
    out: &mut Vec<(String, String, usize, McpServerConfig)>,
) {
    let Ok(entries) = std::fs::read_dir(cache) else {
        return;
    };
    let mut order = 0usize;
    for marketplace in entries.flatten() {
        let Ok(name_entries) = std::fs::read_dir(marketplace.path()) else {
            continue;
        };
        for plugin in name_entries.flatten() {
            let Ok(version_entries) = std::fs::read_dir(plugin.path()) else {
                continue;
            };
            for version in version_entries.flatten() {
                let root = version.path();
                let manifest = root.join(".codex-plugin/plugin.json");
                if !manifest.is_file() {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(&manifest) else {
                    continue;
                };
                let Ok(raw) = serde_json::from_str::<serde_json::Value>(&text) else {
                    continue;
                };
                let mcp_ref = raw.get("mcpServers").or_else(|| raw.get("mcp_servers"));
                let Some(mcp_ref) = mcp_ref.and_then(|v| v.as_str()) else {
                    continue;
                };
                let mcp_path = root.join(mcp_ref);
                let Ok(contents) = std::fs::read_to_string(&mcp_path) else {
                    continue;
                };
                let plugin_id = format!(
                    "{}@{}",
                    plugin.file_name().to_string_lossy(),
                    marketplace.file_name().to_string_lossy()
                );
                if let Ok(outcome) =
                    parse_plugin_mcp_config(&root, &contents, PluginMcpServerPlacement::Declared)
                {
                    for (server_name, config) in outcome.servers {
                        out.push((server_name, plugin_id.clone(), order, config));
                        order += 1;
                    }
                }
            }
        }
    }
}

pub type SharedMcpRuntime = Arc<tokio::sync::RwLock<Option<McpRuntime>>>;

pub fn new_shared_mcp_runtime() -> SharedMcpRuntime {
    Arc::new(tokio::sync::RwLock::new(None))
}

pub async fn refresh_mcp_runtime(slot: &SharedMcpRuntime) {
    {
        let mut guard = slot.write().await;
        if let Some(old) = guard.take() {
            old.manager.shutdown().await;
        }
    }
    let runtime = McpRuntime::from_codex_home(true).await;
    let mut guard = slot.write().await;
    *guard = Some(runtime);
}
