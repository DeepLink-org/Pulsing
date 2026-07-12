//! Shared subprocess execution for Codex shell tools and legacy `Bash`.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use tokio::process::Command;
use tokio::time::timeout;

use crate::approval::{
    args_dangerously_disable_sandbox, effective_sandbox_policy, ensure_shell_allowed,
};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::exec_output::{SHELL_MAX_BYTES, shell_timeout_ms};
use crate::handlers::write::resolve_within_cwd;
use crate::patch::{MaybeApplyPatch, apply_parsed_patch, maybe_parse_apply_patch};
use crate::result::ToolResult;
use crate::sandbox::build_bash_exec;

pub(crate) fn resolve_shell_workdir(
    ctx: &ToolCallContext,
    args: &Value,
) -> Result<PathBuf, ToolError> {
    match args
        .get("workdir")
        .or_else(|| args.get("cwd"))
        .and_then(|v| v.as_str())
    {
        Some(w) => resolve_within_cwd(&ctx.cwd, w).map_err(ToolError::respond),
        None => Ok(ctx.cwd.clone()),
    }
}

pub(crate) async fn run_shell(
    ctx: &ToolCallContext,
    args: &Value,
) -> Result<ToolResult, ToolError> {
    let cmd = args
        .get("cmd")
        .or_else(|| args.get("command"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::respond("missing cmd/command"))?;
    let cwd = resolve_shell_workdir(ctx, args)?;
    let login = args.get("login").and_then(|v| v.as_bool()).unwrap_or(false);
    let timeout_ms = shell_timeout_ms(args);

    ensure_shell_allowed(ctx, args, cmd)?;
    let policy = effective_sandbox_policy(ctx, args);
    let dangerous = args_dangerously_disable_sandbox(ctx, args);

    let plan = build_bash_exec(cmd, Some(&cwd), policy, dangerous, login);

    match maybe_parse_apply_patch(&plan.argv) {
        MaybeApplyPatch::Body(patch_args) => {
            return Ok(match apply_parsed_patch(&patch_args, &cwd) {
                Ok(summary) => ToolResult::ok(summary),
                Err(e) => ToolResult::err(e),
            });
        }
        MaybeApplyPatch::ImplicitInvocation => {
            return Ok(ToolResult::err(
                "patch detected without explicit apply_patch invocation; use apply_patch tool",
            ));
        }
        MaybeApplyPatch::ShellParseError(e) => {
            return Ok(ToolResult::err(format!("shell parse error: {e}")));
        }
        MaybeApplyPatch::PatchParseError(e) => {
            return Ok(ToolResult::err(e.to_string()));
        }
        MaybeApplyPatch::NotApplyPatch => {}
    }

    let mut command = Command::new(&plan.argv[0]);
    command.args(&plan.argv[1..]);
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .current_dir(&cwd);
    if let Some(env) = &plan.env {
        command.env_clear();
        for (k, v) in env {
            command.env(k, v);
        }
    }

    let dur = Duration::from_millis(timeout_ms);
    let out = match timeout(dur, command.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Ok(ToolResult::err(e.to_string())),
        Err(_) => return Ok(ToolResult::err(format!("timed out after {timeout_ms}ms"))),
    };
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&out.stdout));
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    if text.len() > SHELL_MAX_BYTES {
        text.truncate(SHELL_MAX_BYTES);
        text.push_str("\n… truncated …");
    }
    text.push_str(&format!(
        "\nexit={}\n[{}]",
        out.status.code().unwrap_or(-1),
        plan.label
    ));
    if out.status.success() {
        Ok(ToolResult::ok(text))
    } else {
        Ok(ToolResult::err(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::{ApprovalCache, new_exec_policy};
    use crate::context::{LocalToolSession, ToolCallContext};
    use crate::discovery::new_tool_catalog;
    use crate::unified_exec::UnifiedExecManager;
    use std::sync::Arc;

    fn test_ctx(cwd: &Path) -> ToolCallContext {
        ToolCallContext::new(
            cwd,
            "off",
            Arc::new(LocalToolSession::default()),
            Arc::new(UnifiedExecManager::new()),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            new_tool_catalog(),
        )
    }

    #[test]
    fn rejects_workdir_escape_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = test_ctx(dir.path());
        let err =
            resolve_shell_workdir(&ctx, &serde_json::json!({"workdir": "../escape"})).unwrap_err();
        assert!(err.to_string().contains("outside working directory"));
    }
}
