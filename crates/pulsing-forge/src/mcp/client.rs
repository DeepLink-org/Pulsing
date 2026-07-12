//! RMCP client wrapper for stdio and streamable HTTP transports.

use std::collections::HashMap;
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, PaginatedRequestParams, ReadResourceRequestParams,
    ReadResourceResult,
};
use rmcp::service::{ClientInitializeError, RoleClient, RunningService};
use rmcp::transport::ConfigureCommandExt;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::child_process::TokioChildProcess;
use tokio::process::Command;
use tokio::sync::Mutex;

use super::DEFAULT_STARTUP_TIMEOUT_SECS;
use super::config::{McpServerConfig, McpServerEnvVar, McpServerTransportConfig};
use super::tools::{ToolFilter, ToolInfo, filter_tools};

#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    #[error("MCP initialize error: {0}")]
    Initialize(#[from] ClientInitializeError),
    #[error("MCP service error: {0}")]
    Service(#[from] rmcp::service::ServiceError),
    #[error("startup timed out after {0:?}")]
    StartupTimeout(Duration),
    #[error("server {server}: {message}")]
    Startup { server: String, message: String },
}

pub struct McpManagedClient {
    pub server_name: String,
    service: RunningService<RoleClient, ()>,
    pub tools: Vec<ToolInfo>,
    pub tool_filter: ToolFilter,
    pub tool_timeout: Duration,
}

impl McpManagedClient {
    pub async fn connect_stdio(
        server_name: String,
        config: &McpServerConfig,
    ) -> Result<Self, McpClientError> {
        let McpServerTransportConfig::Stdio {
            command,
            args,
            env,
            env_vars,
            cwd,
        } = &config.transport
        else {
            return Err(McpClientError::Startup {
                server: server_name,
                message: "expected stdio transport".into(),
            });
        };

        let transport = TokioChildProcess::new(Command::new(command).configure(|cmd| {
            cmd.args(args);
            if let Some(cwd) = cwd {
                cmd.current_dir(cwd);
            }
            apply_env(cmd, env.as_ref(), env_vars);
        }))
        .map_err(|e| McpClientError::Startup {
            server: server_name.clone(),
            message: e.to_string(),
        })?;

        Self::connect_with_transport(server_name, config, transport).await
    }

    pub async fn connect_streamable_http(
        server_name: String,
        config: &McpServerConfig,
    ) -> Result<Self, McpClientError> {
        let McpServerTransportConfig::StreamableHttp { url, .. } = &config.transport else {
            return Err(McpClientError::Startup {
                server: server_name,
                message: "expected streamable_http transport".into(),
            });
        };

        let transport = StreamableHttpClientTransport::from_uri(url.clone());
        Self::connect_with_transport(server_name, config, transport).await
    }

    async fn connect_with_transport<T>(
        server_name: String,
        config: &McpServerConfig,
        transport: T,
    ) -> Result<Self, McpClientError>
    where
        T: rmcp::transport::Transport<RoleClient> + Send + 'static,
    {
        let timeout = config
            .startup_timeout_sec
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(DEFAULT_STARTUP_TIMEOUT_SECS));

        let connect = async { ().serve(transport).await };

        let service = tokio::time::timeout(timeout, connect)
            .await
            .map_err(|_| McpClientError::StartupTimeout(timeout))??;

        let tool_filter = ToolFilter::from_config(config);
        let listed = service.list_all_tools().await?;
        let mut tools = Vec::new();
        for tool in listed {
            let description = tool.description.as_ref().map(|d| d.to_string());
            tools.push(ToolInfo {
                server_name: server_name.clone(),
                supports_parallel_tool_calls: config.supports_parallel_tool_calls,
                server_origin: Some("plugin".into()),
                callable_name: tool.name.to_string(),
                callable_namespace: server_name.clone(),
                namespace_description: description,
                tool,
                connector_id: None,
                connector_name: None,
                plugin_display_names: vec![],
            });
        }
        tools = filter_tools(tools, &tool_filter);

        Ok(Self {
            server_name,
            service,
            tools,
            tool_filter,
            tool_timeout: config.tool_timeout(),
        })
    }

    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpClientError> {
        let args: rmcp::model::JsonObject = arguments
            .as_object()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        let params = CallToolRequestParams::new(tool_name.to_string()).with_arguments(args);
        let call = self.service.call_tool(params);
        tokio::time::timeout(self.tool_timeout, call)
            .await
            .map_err(|_| McpClientError::Startup {
                server: self.server_name.clone(),
                message: format!("tool call timed out after {:?}", self.tool_timeout),
            })?
            .map_err(McpClientError::Service)
    }

    pub async fn list_resources(
        &self,
        params: Option<PaginatedRequestParams>,
    ) -> Result<rmcp::model::ListResourcesResult, McpClientError> {
        self.service
            .list_resources(params)
            .await
            .map_err(McpClientError::Service)
    }

    pub async fn list_resource_templates(
        &self,
        params: Option<PaginatedRequestParams>,
    ) -> Result<rmcp::model::ListResourceTemplatesResult, McpClientError> {
        self.service
            .list_resource_templates(params)
            .await
            .map_err(McpClientError::Service)
    }

    pub async fn read_resource(
        &self,
        params: ReadResourceRequestParams,
    ) -> Result<ReadResourceResult, McpClientError> {
        self.service
            .read_resource(params)
            .await
            .map_err(McpClientError::Service)
    }

    pub async fn close_connection(&mut self) {
        let _ = self.service.close().await;
    }

    pub async fn shutdown(mut self) -> Result<(), McpClientError> {
        self.close_connection().await;
        Ok(())
    }
}

fn apply_env(
    cmd: &mut Command,
    env: Option<&HashMap<String, String>>,
    env_vars: &[McpServerEnvVar],
) {
    if let Some(env) = env {
        for (k, v) in env {
            cmd.env(k, v);
        }
    }
    for var in env_vars {
        if let Ok(val) = std::env::var(var.name()) {
            cmd.env(var.name(), val);
        }
    }
}

pub type SharedMcpClient = std::sync::Arc<Mutex<McpManagedClient>>;
