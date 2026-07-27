use serde_json::Value;

use super::shell_exec::run_shell;
use crate::context::ToolCallContext;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

pub struct ShellCommandHandler;

impl ToolExecutor for ShellCommandHandler {
    fn tool_name(&self) -> &str {
        "shell_command"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { run_shell(ctx, &arguments).await })
    }
}

pub struct ExecCommandHandler;

impl ToolExecutor for ExecCommandHandler {
    fn tool_name(&self) -> &str {
        "exec_command"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { ctx.exec.exec_command(ctx, &arguments).await })
    }
}

pub struct WriteStdinHandler;

impl ToolExecutor for WriteStdinHandler {
    fn tool_name(&self) -> &str {
        "write_stdin"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { ctx.exec.write_stdin(ctx, &arguments).await })
    }
}
