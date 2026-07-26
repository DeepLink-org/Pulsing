use serde_json::Value;

use super::{err, ok};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};
use crate::patch::{MaybeApplyPatch, apply_parsed_patch, parse_patch};

pub struct ApplyPatchHandler;

impl ToolExecutor for ApplyPatchHandler {
    fn tool_name(&self) -> &str {
        "apply_patch"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        let cwd = ctx.cwd.clone();
        Box::pin(async move { apply_patch_impl(&cwd, arguments) })
    }
}

fn apply_patch_impl(
    cwd: &std::path::Path,
    arguments: Value,
) -> Result<crate::result::ToolResult, ToolError> {
    if let Some(raw) = arguments.as_str() {
        return apply_patch_text(raw, cwd);
    }
    if let Some(patch) = arguments.get("patch").and_then(|v| v.as_str()) {
        return apply_patch_text(patch, cwd);
    }
    if let Some(input) = arguments.get("input").and_then(|v| v.as_str()) {
        return apply_patch_text(input, cwd);
    }
    if let Some(cmd) = arguments.get("command").and_then(|v| v.as_array()) {
        let argv: Vec<String> = cmd
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        return apply_patch_argv(&argv, cwd);
    }
    Err(ToolError::respond(
        "apply_patch expects patch text, {\"patch\": \"...\"}, or {\"command\": [\"apply_patch\", \"...\"]}",
    ))
}

fn apply_patch_text(
    patch: &str,
    cwd: &std::path::Path,
) -> Result<crate::result::ToolResult, ToolError> {
    let args = parse_patch(patch).map_err(|e| ToolError::respond(e.to_string()))?;
    match apply_parsed_patch(&args, cwd) {
        Ok(summary) => ok(summary),
        Err(e) => err(e),
    }
}

fn apply_patch_argv(
    argv: &[String],
    cwd: &std::path::Path,
) -> Result<crate::result::ToolResult, ToolError> {
    match crate::patch::maybe_parse_apply_patch(argv) {
        MaybeApplyPatch::Body(args) => match apply_parsed_patch(&args, cwd) {
            Ok(summary) => ok(summary),
            Err(e) => err(e),
        },
        MaybeApplyPatch::ImplicitInvocation => {
            err("patch detected without explicit apply_patch invocation; use apply_patch tool")
        }
        MaybeApplyPatch::PatchParseError(e) => err(e.to_string()),
        MaybeApplyPatch::ShellParseError(e) => err(e),
        MaybeApplyPatch::NotApplyPatch => err("not an apply_patch command"),
    }
}
