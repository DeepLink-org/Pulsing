use serde_json::Value;

use super::shell_exec::run_shell;
use crate::context::ToolCallContext;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

/// Legacy Claude-style alias; Codex uses `shell_command` / `exec_command`.
pub struct BashHandler;

impl ToolExecutor for BashHandler {
    fn tool_name(&self) -> &str {
        "Bash"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { run_shell(ctx, &arguments).await })
    }
}
