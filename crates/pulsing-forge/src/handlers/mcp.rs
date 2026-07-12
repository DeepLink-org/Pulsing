//! MCP resource + dynamic tool handlers (Codex-aligned).

use serde_json::Value;

use rmcp::model::{PaginatedRequestParams, ReadResourceRequestParams};

use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture, ToolExposure};
use crate::mcp::{
    LEGACY_MCP_TOOL_NAME_PREFIX, McpRuntime, enforce_resource_size_limit, validate_mcp_resource_uri,
};
use crate::mcp::{McpClientError, ToolInfo};
use crate::result::ToolResult;

fn optional_server_name(arguments: &Value) -> Result<Option<String>, ToolError> {
    match arguments.get("server") {
        None => Ok(None),
        Some(v) => {
            let raw = v
                .as_str()
                .ok_or_else(|| ToolError::respond("server must be a string"))?;
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Err(ToolError::respond("server must be a non-empty string"));
            }
            Ok(Some(trimmed.to_string()))
        }
    }
}

fn require_non_empty_str<'a>(arguments: &'a Value, field: &str) -> Result<&'a str, ToolError> {
    let raw = arguments
        .get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::respond(format!("{field} is required")))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(ToolError::respond(format!(
            "{field} must be a non-empty string"
        )));
    }
    Ok(trimmed)
}

macro_rules! mcp_handler {
    ($ty:ident, $tool:literal) => {
        pub struct $ty;

        impl ToolExecutor for $ty {
            fn tool_name(&self) -> &str {
                $tool
            }

            fn supports_parallel(&self) -> bool {
                true
            }

            fn handle<'a>(
                &'a self,
                ctx: &'a ToolCallContext,
                arguments: Value,
            ) -> ToolExecutorFuture<'a> {
                Box::pin(async move {
                    let rt = ctx
                        .mcp_runtime
                        .as_ref()
                        .ok_or_else(|| ToolError::respond("MCP runtime is not initialized"))?;
                    let guard = rt.read().await;
                    let mcp = guard
                        .as_ref()
                        .ok_or_else(|| ToolError::respond("MCP runtime is not started"))?;
                    dispatch_mcp_host_tool($tool, mcp, arguments).await
                })
            }
        }
    };
}

mcp_handler!(ListMcpResourcesHandler, "list_mcp_resources");
mcp_handler!(
    ListMcpResourceTemplatesHandler,
    "list_mcp_resource_templates"
);
mcp_handler!(ReadMcpResourceHandler, "read_mcp_resource");

pub struct McpDynamicToolHandler {
    pub model_name: String,
    pub supports_parallel: bool,
}

impl McpDynamicToolHandler {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            supports_parallel: false,
        }
    }

    pub fn from_tool_info(info: &ToolInfo, prefix_mcp_tool_names: bool) -> Self {
        Self {
            model_name: info.model_tool_name(prefix_mcp_tool_names),
            supports_parallel: info.supports_parallel_tool_calls,
        }
    }
}

impl ToolExecutor for McpDynamicToolHandler {
    fn tool_name(&self) -> &str {
        &self.model_name
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Deferred
    }

    fn supports_parallel(&self) -> bool {
        self.supports_parallel
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        let name = self.model_name.clone();
        Box::pin(async move {
            let rt = ctx
                .mcp_runtime
                .as_ref()
                .ok_or_else(|| ToolError::respond("MCP runtime is not initialized"))?;
            let guard = rt.read().await;
            let mcp = guard
                .as_ref()
                .ok_or_else(|| ToolError::respond("MCP runtime is not started"))?;
            dispatch_mcp_dynamic_tool(mcp, &name, arguments).await
        })
    }
}

/// Dispatch an MCP function tool by model-visible name. Used by [`McpDynamicToolHandler`]
/// and [`ToolRuntime::call_tool`] fall-through.
pub async fn dispatch_mcp_dynamic_tool(
    mcp: &McpRuntime,
    model_name: &str,
    arguments: Value,
) -> Result<ToolResult, ToolError> {
    let tool = mcp.manager.find_tool(model_name).ok_or_else(|| {
        ToolError::respond(format!(
            "unknown MCP tool {model_name:?}; refresh MCP or list names via mcp_tool_names()"
        ))
    })?;
    let result = mcp
        .manager
        .call_tool_by_model_name(model_name, arguments)
        .await
        .map_err(|e| format_mcp_dynamic_tool_error(&tool.server_name, &tool.tool.name, e))?;
    Ok(format_call_tool_result(result))
}

/// Returns `Some` when `name` is a registered MCP dynamic tool, or uses the
/// ``mcp__`` prefix while MCP runtime is started (unknown names get MCP errors).
pub async fn try_call_mcp_dynamic_tool(
    ctx: &ToolCallContext,
    name: &str,
    arguments: Value,
) -> Option<Result<ToolResult, ToolError>> {
    let slot = ctx.mcp_runtime.as_ref()?;
    let guard = slot.read().await;
    let mcp = guard.as_ref()?;
    if name.starts_with(LEGACY_MCP_TOOL_NAME_PREFIX) || mcp.manager.find_tool(name).is_some() {
        return Some(dispatch_mcp_dynamic_tool(mcp, name, arguments).await);
    }
    None
}

fn format_mcp_dynamic_tool_error(server: &str, tool: &str, err: McpClientError) -> ToolError {
    ToolError::respond(format!("MCP tool {server}/{tool} failed: {err}"))
}

fn format_mcp_read_resource_error(server: &str, uri: &str, err: McpClientError) -> ToolError {
    ToolError::respond(format!("MCP resource {server}/{uri} failed: {err}"))
}

async fn dispatch_mcp_host_tool(
    tool: &str,
    mcp: &McpRuntime,
    arguments: Value,
) -> Result<ToolResult, ToolError> {
    match tool {
        "list_mcp_resources" => {
            let server = optional_server_name(&arguments)?;
            let cursor = arguments
                .get("cursor")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            if let Some(server_name) = server {
                let params = cursor.map(|c| PaginatedRequestParams::default().with_cursor(Some(c)));
                let resources = mcp
                    .manager
                    .list_resources(&server_name, params)
                    .await
                    .map_err(|e| ToolError::respond(e.to_string()))?;
                Ok(ToolResult::ok(
                    serde_json::to_string(&resources).unwrap_or_default(),
                ))
            } else {
                if cursor.is_some() {
                    return Err(ToolError::respond(
                        "cursor can only be used when a server is specified",
                    ));
                }
                let all = mcp.manager.list_all_resources().await;
                Ok(ToolResult::ok(
                    serde_json::to_string(&all).unwrap_or_default(),
                ))
            }
        }
        "list_mcp_resource_templates" => {
            let server = require_non_empty_str(&arguments, "server")?;
            let cursor = arguments
                .get("cursor")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let params = cursor.map(|c| PaginatedRequestParams::default().with_cursor(Some(c)));
            let templates = mcp
                .manager
                .list_resource_templates(server, params)
                .await
                .map_err(|e| ToolError::respond(e.to_string()))?;
            Ok(ToolResult::ok(
                serde_json::to_string(&templates).unwrap_or_default(),
            ))
        }
        "read_mcp_resource" => {
            let server = require_non_empty_str(&arguments, "server")?;
            let uri = require_non_empty_str(&arguments, "uri")?;
            validate_mcp_resource_uri(uri).map_err(ToolError::respond)?;
            let params = ReadResourceRequestParams::new(uri.to_string());
            let result = mcp
                .manager
                .read_resource(server, params)
                .await
                .map_err(|e| format_mcp_read_resource_error(server, uri, e))?;
            let result = enforce_resource_size_limit(result)
                .map_err(|e| ToolError::respond(format!("MCP resource {server}/{uri}: {e}")))?;
            Ok(ToolResult::ok(
                serde_json::to_string(&result).unwrap_or_default(),
            ))
        }
        _ => Err(ToolError::respond(format!("unknown MCP host tool: {tool}"))),
    }
}

fn format_call_tool_result(result: rmcp::model::CallToolResult) -> ToolResult {
    let mut out = ToolResult::ok(serde_json::to_string(&result).unwrap_or_default());
    out.is_error = result.is_error.unwrap_or(false);
    out
}

pub fn mcp_resource_handlers() -> Vec<Box<dyn ToolExecutor>> {
    vec![
        Box::new(ListMcpResourcesHandler),
        Box::new(ListMcpResourceTemplatesHandler),
        Box::new(ReadMcpResourceHandler),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::{McpConnectionManager, McpRuntime, build_default_catalog};

    async fn empty_runtime() -> McpRuntime {
        let catalog = build_default_catalog(vec![], std::collections::HashMap::new());
        let manager = McpConnectionManager::start(&catalog, true).await;
        McpRuntime { catalog, manager }
    }

    #[tokio::test]
    async fn dynamic_tool_unknown_name() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_dynamic_tool(
            &mcp,
            "mcp__missing__tool",
            Value::Object(Default::default()),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("unknown MCP tool"));
    }

    #[tokio::test]
    async fn read_mcp_resource_requires_server() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "read_mcp_resource",
            &mcp,
            serde_json::json!({"uri": "file:///tmp/x"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("server is required"));
    }

    #[tokio::test]
    async fn read_mcp_resource_rejects_invalid_uri() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "read_mcp_resource",
            &mcp,
            serde_json::json!({"server": "demo", "uri": "not-a-uri"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("invalid uri"));
    }

    #[tokio::test]
    async fn read_mcp_resource_rejects_empty_server() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "read_mcp_resource",
            &mcp,
            serde_json::json!({"server": "  ", "uri": "file:///tmp/x"}),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("server must be a non-empty string")
        );
    }

    #[tokio::test]
    async fn read_mcp_resource_requires_uri() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "read_mcp_resource",
            &mcp,
            serde_json::json!({"server": "demo"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("uri is required"));
    }

    #[tokio::test]
    async fn read_mcp_resource_rejects_empty_uri() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "read_mcp_resource",
            &mcp,
            serde_json::json!({"server": "demo", "uri": "   "}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("uri must be a non-empty string"));
    }

    #[tokio::test]
    async fn read_mcp_resource_unknown_server_includes_context() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "read_mcp_resource",
            &mcp,
            serde_json::json!({"server": "missing", "uri": "file:///tmp/x"}),
        )
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("MCP resource missing/file:///tmp/x failed"));
        assert!(msg.contains("MCP server not connected"));
    }

    #[tokio::test]
    async fn list_mcp_resource_templates_requires_server() {
        let mcp = empty_runtime().await;
        let err =
            dispatch_mcp_host_tool("list_mcp_resource_templates", &mcp, serde_json::json!({}))
                .await
                .unwrap_err();
        assert!(err.to_string().contains("server is required"));
    }

    #[tokio::test]
    async fn list_mcp_resource_templates_rejects_empty_server() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "list_mcp_resource_templates",
            &mcp,
            serde_json::json!({"server": "  "}),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("server must be a non-empty string")
        );
    }

    #[tokio::test]
    async fn list_mcp_resource_templates_unknown_server() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "list_mcp_resource_templates",
            &mcp,
            serde_json::json!({"server": "missing"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("MCP server not connected"));
    }

    #[tokio::test]
    async fn list_mcp_resources_empty_catalog_returns_object() {
        let mcp = empty_runtime().await;
        let out = dispatch_mcp_host_tool("list_mcp_resources", &mcp, serde_json::json!({}))
            .await
            .expect("list all resources");
        assert!(!out.is_error);
        assert_eq!(out.content, "{}");
    }

    #[tokio::test]
    async fn list_mcp_resources_cursor_requires_server() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "list_mcp_resources",
            &mcp,
            serde_json::json!({"cursor": "next"}),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("cursor can only be used when a server is specified")
        );
    }

    #[tokio::test]
    async fn list_mcp_resources_unknown_server_errors() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "list_mcp_resources",
            &mcp,
            serde_json::json!({"server": "missing-server"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("MCP server not connected"));
    }

    #[tokio::test]
    async fn list_mcp_resources_rejects_blank_server() {
        let mcp = empty_runtime().await;
        let err = dispatch_mcp_host_tool(
            "list_mcp_resources",
            &mcp,
            serde_json::json!({"server": "   "}),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("server must be a non-empty string")
        );
    }

    #[tokio::test]
    async fn list_mcp_resources_rejects_non_string_server() {
        let mcp = empty_runtime().await;
        let err =
            dispatch_mcp_host_tool("list_mcp_resources", &mcp, serde_json::json!({"server": 1}))
                .await
                .unwrap_err();
        assert!(err.to_string().contains("server must be a string"));
    }
}
