//! Built-in tool handlers.

mod apply_patch;
mod bash;
mod discovery;
mod edit;
mod glob;
mod grep;
mod plan;
mod read;
mod request_permissions;
mod session;
mod shell;
pub(crate) mod shell_exec;
mod view_image;
mod write;

mod mcp;

pub use mcp::{
    ListMcpResourceTemplatesHandler, ListMcpResourcesHandler, McpDynamicToolHandler,
    ReadMcpResourceHandler, dispatch_mcp_dynamic_tool, mcp_resource_handlers,
    try_call_mcp_dynamic_tool,
};

pub use apply_patch::ApplyPatchHandler;
pub use bash::BashHandler;
pub use discovery::{ListAvailablePluginsHandler, RequestPluginInstallHandler, ToolSearchHandler};
pub use edit::EditHandler;
pub use glob::GlobHandler;
pub use grep::GrepHandler;
pub use plan::UpdatePlanHandler;
pub use read::ReadHandler;
pub use request_permissions::RequestPermissionsHandler;
pub use session::{
    GetContextRemainingHandler, NEW_CONTEXT_MESSAGE, NewContextHandler, RequestUserInputHandler,
};
pub use shell::{ExecCommandHandler, ShellCommandHandler, WriteStdinHandler};
pub use view_image::ViewImageHandler;
pub use write::WriteHandler;

use crate::error::ToolError;
use crate::executor::ToolExecutor;
use crate::result::ToolResult;

pub fn builtin_handlers() -> Vec<Box<dyn ToolExecutor>> {
    vec![
        // Codex shell / file
        Box::new(ShellCommandHandler),
        Box::new(ExecCommandHandler),
        Box::new(WriteStdinHandler),
        Box::new(ApplyPatchHandler),
        Box::new(ViewImageHandler),
        // Codex plan / session
        Box::new(UpdatePlanHandler),
        Box::new(NewContextHandler),
        Box::new(GetContextRemainingHandler),
        Box::new(RequestUserInputHandler),
        Box::new(RequestPermissionsHandler),
        Box::new(ToolSearchHandler),
        Box::new(ListAvailablePluginsHandler),
        Box::new(RequestPluginInstallHandler),
        // MCP resource helpers (Codex core tools)
        Box::new(ListMcpResourcesHandler),
        Box::new(ListMcpResourceTemplatesHandler),
        Box::new(ReadMcpResourceHandler),
        // Claude-style helpers (kept for Craft compatibility)
        Box::new(ReadHandler),
        Box::new(GlobHandler),
        Box::new(GrepHandler),
        Box::new(EditHandler),
        Box::new(WriteHandler),
        Box::new(BashHandler),
    ]
}

pub(crate) fn json_str<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a str, ToolError> {
    v.get(key)
        .and_then(|x| x.as_str())
        .ok_or_else(|| ToolError::respond(format!("missing or invalid string field {key:?}")))
}

pub(crate) fn ok(content: impl Into<String>) -> Result<ToolResult, ToolError> {
    Ok(ToolResult::ok(content))
}

pub(crate) fn err(content: impl Into<String>) -> Result<ToolResult, ToolError> {
    Ok(ToolResult::err(content))
}
