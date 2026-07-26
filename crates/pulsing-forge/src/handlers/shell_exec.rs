//! Shared subprocess execution for Codex shell tools and legacy `Bash`.

use std::path::PathBuf;
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
use crate::process_group::{ProcessGroupGuard, configure};
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
    if ctx.turn.as_ref().is_some_and(|turn| turn.is_cancelled()) {
        return Ok(ToolResult::err("shell command cancelled before start"));
    }

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
    configure(&mut command);
    command.kill_on_drop(true);
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

    let _turn_resource = ctx
        .turn
        .as_ref()
        .map(|turn| turn.resources().register_passive("shell_command"));
    let dur = Duration::from_millis(timeout_ms);
    let child = match command.spawn() {
        Ok(child) => child,
        Err(err) => return Ok(ToolResult::err(err.to_string())),
    };
    let mut process_group = ProcessGroupGuard::new(child.id());
    let output = child.wait_with_output();
    tokio::pin!(output);
    let result = if let Some(turn) = &ctx.turn {
        let cancellation = turn.cancellation();
        tokio::select! {
            _ = cancellation.cancelled() => {
                process_group.kill_now();
                let _ = timeout(Duration::from_secs(2), &mut output).await;
                return Ok(ToolResult::err("shell command cancelled"));
            }
            result = timeout(dur, &mut output) => result,
        }
    } else {
        timeout(dur, &mut output).await
    };
    let out = match result {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Ok(ToolResult::err(e.to_string())),
        Err(_) => {
            process_group.kill_now();
            let _ = timeout(Duration::from_secs(2), &mut output).await;
            return Ok(ToolResult::err(format!("timed out after {timeout_ms}ms")));
        }
    };
    process_group.disarm();
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
    use crate::approval::{ApprovalCache, ApprovalPolicy, new_exec_policy};
    use crate::context::{LocalToolSession, ToolCallContext};
    use crate::discovery::new_tool_catalog;
    use crate::unified_exec::UnifiedExecManager;
    use std::path::Path;
    use std::sync::Arc;

    fn test_ctx(cwd: &Path) -> ToolCallContext {
        ToolCallContext::new(
            cwd,
            "off",
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always)),
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

    #[tokio::test]
    async fn turn_cancellation_kills_the_shell_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("should-not-exist");
        let turn = Arc::new(crate::turn::TurnExecutionContext::new(
            crate::SessionId::new(),
            crate::TurnId::new(),
        ));
        let ctx = test_ctx(dir.path()).with_turn(turn.clone());
        let args = serde_json::json!({
            "cmd": "(sleep 0.3; touch should-not-exist) & wait",
            "timeout_ms": 5_000
        });

        let cancel = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            turn.cancel();
        };
        let (result, ()) = tokio::join!(run_shell(&ctx, &args), cancel);
        assert!(result.unwrap().is_error);
        assert!(turn.resources().wait_for_idle(Duration::from_secs(1)).await);
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(!marker.exists(), "cancelled process tree produced a file");
    }
}
