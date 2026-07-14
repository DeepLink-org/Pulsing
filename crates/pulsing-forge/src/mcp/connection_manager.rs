//! Aggregates MCP server connections (Codex `McpConnectionManager` subset).

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::model::{
    CallToolResult, PaginatedRequestParams, ReadResourceRequestParams, ReadResourceResult,
    Resource, ResourceTemplate,
};
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::catalog::ResolvedMcpCatalog;
use super::client::{McpClientError, McpManagedClient};
use super::config::McpServerTransportConfig;
use super::tools::{ToolInfo, normalize_tools_for_model, tool_is_model_visible};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpStartupStatus {
    Starting,
    Ready,
    Failed,
}

#[derive(Clone, Debug)]
pub struct McpStartupFailure {
    pub server: String,
    pub message: String,
}

pub struct McpConnectionManager {
    clients: HashMap<String, Arc<Mutex<McpManagedClient>>>,
    all_tools: Vec<ToolInfo>,
    prefix_mcp_tool_names: bool,
    failures: Vec<McpStartupFailure>,
}

impl McpConnectionManager {
    pub async fn start(catalog: &ResolvedMcpCatalog, prefix_mcp_tool_names: bool) -> Self {
        let mut clients = HashMap::new();
        let mut failures = Vec::new();
        let mut join = tokio::task::JoinSet::new();

        for (name, server) in &catalog.servers {
            if !server.config.enabled {
                continue;
            }
            let name = name.clone();
            let config = server.config.clone();
            join.spawn(async move {
                let result = match &config.transport {
                    McpServerTransportConfig::Stdio { .. } => {
                        McpManagedClient::connect_stdio(name.clone(), &config).await
                    }
                    McpServerTransportConfig::StreamableHttp { .. } => {
                        McpManagedClient::connect_streamable_http(name.clone(), &config).await
                    }
                };
                (name, result)
            });
        }

        while let Some(res) = join.join_next().await {
            match res {
                Ok((name, Ok(client))) => {
                    info!(server = %name, tools = client.tools.len(), "MCP server ready");
                    clients.insert(name, Arc::new(Mutex::new(client)));
                }
                Ok((name, Err(err))) => {
                    warn!(server = %name, error = %err, "MCP server startup failed");
                    failures.push(McpStartupFailure {
                        server: name,
                        message: err.to_string(),
                    });
                }
                Err(err) => {
                    warn!(error = %err, "MCP startup task join failed");
                }
            }
        }

        let mut all_tools = Vec::new();
        for client in clients.values() {
            let guard = client.lock().await;
            for tool in &guard.tools {
                if tool_is_model_visible(tool) {
                    all_tools.push(tool.clone());
                }
            }
        }
        all_tools = normalize_tools_for_model(all_tools, prefix_mcp_tool_names);

        Self {
            clients,
            all_tools,
            prefix_mcp_tool_names,
            failures,
        }
    }

    pub fn failures(&self) -> &[McpStartupFailure] {
        &self.failures
    }

    pub fn list_all_tools(&self) -> &[ToolInfo] {
        &self.all_tools
    }

    pub fn find_tool(&self, model_name: &str) -> Option<&ToolInfo> {
        self.all_tools
            .iter()
            .find(|t| t.model_tool_name(self.prefix_mcp_tool_names) == model_name)
    }

    pub async fn call_tool_by_model_name(
        &self,
        model_name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpClientError> {
        let tool = self.find_tool(model_name).ok_or_else(|| {
            let hint = if self.all_tools.is_empty() {
                "no MCP tools are registered (check server startup failures)".into()
            } else {
                format!(
                    "available tools: {}",
                    self.all_tools
                        .iter()
                        .map(|t| t.model_tool_name(self.prefix_mcp_tool_names))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            McpClientError::Startup {
                server: model_name.into(),
                message: format!("unknown MCP tool; {hint}"),
            }
        })?;
        let client =
            self.clients
                .get(&tool.server_name)
                .ok_or_else(|| McpClientError::Startup {
                    server: tool.server_name.clone(),
                    message: format!("MCP server not connected for tool {:?}", tool.tool.name),
                })?;
        client
            .lock()
            .await
            .call_tool(&tool.tool.name, arguments)
            .await
    }

    pub async fn call_tool_raw(
        &self,
        server: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpClientError> {
        let client = self
            .clients
            .get(server)
            .ok_or_else(|| McpClientError::Startup {
                server: server.into(),
                message: "MCP server not connected".into(),
            })?;
        client.lock().await.call_tool(tool_name, arguments).await
    }

    pub async fn list_resources(
        &self,
        server: &str,
        params: Option<PaginatedRequestParams>,
    ) -> Result<Vec<Resource>, McpClientError> {
        let client = self
            .clients
            .get(server)
            .ok_or_else(|| McpClientError::Startup {
                server: server.into(),
                message: "MCP server not connected".into(),
            })?;
        Ok(client.lock().await.list_resources(params).await?.resources)
    }

    pub async fn list_all_resources(&self) -> HashMap<String, Vec<Resource>> {
        let mut out = HashMap::new();
        for (name, client) in &self.clients {
            if let Ok(result) = client.lock().await.list_resources(None).await {
                out.insert(name.clone(), result.resources);
            }
        }
        out
    }

    /// Best-effort close of all live MCP transports (used before refresh).
    pub async fn shutdown(&self) {
        for client in self.clients.values() {
            let mut guard = client.lock().await;
            guard.close_connection().await;
        }
    }

    pub async fn list_resource_templates(
        &self,
        server: &str,
        params: Option<PaginatedRequestParams>,
    ) -> Result<Vec<ResourceTemplate>, McpClientError> {
        let client = self
            .clients
            .get(server)
            .ok_or_else(|| McpClientError::Startup {
                server: server.into(),
                message: "MCP server not connected".into(),
            })?;
        Ok(client
            .lock()
            .await
            .list_resource_templates(params)
            .await?
            .resource_templates)
    }

    pub async fn read_resource(
        &self,
        server: &str,
        params: ReadResourceRequestParams,
    ) -> Result<ReadResourceResult, McpClientError> {
        let client = self
            .clients
            .get(server)
            .ok_or_else(|| McpClientError::Startup {
                server: server.into(),
                message: "MCP server not connected".into(),
            })?;
        client.lock().await.read_resource(params).await
    }

    pub fn server_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.clients.keys().cloned().collect();
        names.sort();
        names
    }
}

#[cfg(test)]
impl McpConnectionManager {
    pub fn from_tools(tools: Vec<ToolInfo>, prefix_mcp_tool_names: bool) -> Self {
        Self {
            clients: HashMap::new(),
            all_tools: normalize_tools_for_model(tools, prefix_mcp_tool_names),
            prefix_mcp_tool_names,
            failures: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::build_default_catalog;

    #[tokio::test]
    async fn list_all_resources_empty_when_no_clients() {
        let catalog = build_default_catalog(vec![], HashMap::new());
        let manager = McpConnectionManager::start(&catalog, true).await;
        let all = manager.list_all_resources().await;
        assert!(all.is_empty());
    }

    #[tokio::test]
    async fn list_resources_unknown_server_errors() {
        let catalog = build_default_catalog(vec![], HashMap::new());
        let manager = McpConnectionManager::start(&catalog, true).await;
        let err = manager.list_resources("missing", None).await.unwrap_err();
        assert!(err.to_string().contains("MCP server not connected"));
    }
}
