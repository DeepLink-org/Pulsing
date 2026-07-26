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
use crate::registry::ToolSpec;
use crate::result::ToolResult;
use serde_json::json;

/// Canonical model schemas for built-in Forge executors.
///
/// Built-in executors explicitly return their entry from this catalog, so both
/// registry exposure and dispatch are rooted in the same registered object.
/// Dynamic executors such as MCP tools provide their own `ToolExecutor::spec`.
pub(crate) fn builtin_spec(name: &str) -> ToolSpec {
    let object = |properties, required: &[&str]| {
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        })
    };
    match name {
        "shell_command" => ToolSpec::function(
            name,
            "Run a shell command in the workspace.",
            object(
                json!({
                    "command": {"type": "string"},
                    "cmd": {"type": "string"},
                    "workdir": {"type": "string"},
                    "timeout_ms": {"type": "integer"},
                    "login": {"type": "boolean"},
                    "sandbox_permissions": {"type": "string"},
                    "justification": {"type": "string"}
                }),
                &[],
            ),
        ),
        "exec_command" => ToolSpec::function(
            name,
            "Start a command, optionally keeping an interactive PTY session.",
            object(
                json!({
                    "cmd": {"type": "string"},
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                    "yield_time_ms": {"type": "integer"},
                    "max_output_tokens": {"type": "integer"},
                    "tty": {"type": "boolean"},
                    "login": {"type": "boolean"},
                    "sandbox_permissions": {"type": "string"},
                    "justification": {"type": "string"},
                    "prefix_rule": {"type": "array", "items": {"type": "string"}}
                }),
                &[],
            ),
        ),
        "write_stdin" => ToolSpec::function(
            name,
            "Write input to or poll a running exec_command session.",
            object(
                json!({
                    "session_id": {"type": "integer"},
                    "chars": {"type": "string"},
                    "yield_time_ms": {"type": "integer"},
                    "max_output_tokens": {"type": "integer"}
                }),
                &["session_id"],
            ),
        ),
        "apply_patch" => ToolSpec::function(
            name,
            "Apply a structured patch to files in the workspace.",
            object(json!({"patch": {"type": "string"}}), &["patch"]),
        ),
        "view_image" => ToolSpec::function(
            name,
            "Load a local image for visual inspection.",
            object(
                json!({
                    "path": {"type": "string"},
                    "detail": {"type": "string", "enum": ["high", "original"]}
                }),
                &["path"],
            ),
        ),
        "update_plan" => ToolSpec::function(
            name,
            "Update the multi-step plan visible to the user.",
            object(
                json!({
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["step", "status"],
                            "additionalProperties": false
                        }
                    },
                    "explanation": {"type": "string"}
                }),
                &["plan"],
            ),
        ),
        "new_context" => ToolSpec::function(
            name,
            "Request a fresh context window for the current Forge session.",
            object(json!({}), &[]),
        ),
        "get_context_remaining" => ToolSpec::function(
            name,
            "Return the estimated tokens remaining in the current context.",
            object(json!({}), &[]),
        ),
        "request_user_input" => ToolSpec::function(
            name,
            "Ask the user one or more short structured questions.",
            object(
                json!({
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "header": {"type": "string"},
                                "question": {"type": "string"},
                                "isOther": {"type": "boolean"},
                                "isSecret": {"type": "boolean"},
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string"},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["label", "description"],
                                        "additionalProperties": false
                                    }
                                }
                            },
                            "required": ["id", "header", "question"],
                            "additionalProperties": false
                        }
                    },
                    "autoResolutionMs": {"type": "integer", "minimum": 60000, "maximum": 240000}
                }),
                &["questions"],
            ),
        ),
        "request_permissions" => ToolSpec::function(
            name,
            "Request scoped network or filesystem permissions from the host.",
            object(
                json!({
                    "permissions": {
                        "type": "object",
                        "properties": {
                            "network": {"type": "object"},
                            "file_system": {"type": "object"}
                        }
                    },
                    "reason": {"type": "string"}
                }),
                &["permissions"],
            ),
        ),
        "tool_search" => ToolSpec::function(
            name,
            "Search deferred tools and integrations available to this session.",
            object(
                json!({
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1}
                }),
                &["query"],
            ),
        ),
        "list_available_plugins_to_install" => ToolSpec::function(
            name,
            "List Forge plugins that are available but not installed.",
            object(json!({}), &[]),
        ),
        "request_plugin_install" => ToolSpec::function(
            name,
            "Request installation of a specific Forge plugin.",
            object(
                json!({
                    "tool_id": {"type": "string"},
                    "tool_type": {"type": "string", "enum": ["plugin", "connector"]},
                    "action_type": {"type": "string", "enum": ["install", "enable"]},
                    "suggest_reason": {"type": "string"}
                }),
                &["tool_id", "suggest_reason"],
            ),
        ),
        "list_mcp_resources" => ToolSpec::function(
            name,
            "List resources exposed by connected MCP servers.",
            object(
                json!({
                    "server": {"type": "string"},
                    "cursor": {"type": "string"}
                }),
                &[],
            ),
        ),
        "list_mcp_resource_templates" => ToolSpec::function(
            name,
            "List resource templates exposed by one MCP server.",
            object(
                json!({
                    "server": {"type": "string"},
                    "cursor": {"type": "string"}
                }),
                &["server"],
            ),
        ),
        "read_mcp_resource" => ToolSpec::function(
            name,
            "Read a resource from a connected MCP server.",
            object(
                json!({
                    "server": {"type": "string"},
                    "uri": {"type": "string"}
                }),
                &["server", "uri"],
            ),
        ),
        "Read" => ToolSpec::function(
            name,
            "Read a UTF-8 file from the workspace.",
            object(
                json!({
                    "file_path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1}
                }),
                &["file_path"],
            ),
        ),
        "Glob" => ToolSpec::function(
            name,
            "Find files by glob pattern.",
            object(
                json!({
                    "pattern": {"type": "string"},
                    "path": {"type": "string"}
                }),
                &["pattern"],
            ),
        ),
        "Grep" => ToolSpec::function(
            name,
            "Search workspace file contents with a regular expression.",
            object(
                json!({
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"}
                }),
                &["pattern"],
            ),
        ),
        "Edit" => ToolSpec::function(
            name,
            "Replace one unique string occurrence in a workspace file.",
            object(
                json!({
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"}
                }),
                &["file_path", "old_string", "new_string"],
            ),
        ),
        "Write" => ToolSpec::function(
            name,
            "Create or overwrite a file inside the workspace.",
            object(
                json!({
                    "file_path": {"type": "string"},
                    "content": {"type": "string"}
                }),
                &["file_path", "content"],
            ),
        ),
        "Bash" => ToolSpec::function(
            name,
            "Legacy alias for running a shell command in the workspace.",
            object(
                json!({
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                    "timeout_ms": {"type": "integer"}
                }),
                &["command"],
            ),
        ),
        _ => panic!("tool {name:?} must implement ToolExecutor::spec"),
    }
}

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
