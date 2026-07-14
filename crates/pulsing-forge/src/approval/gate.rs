//! Gate shell / unified-exec behind execpolicy + host approval.

use serde_json::Value;

use crate::approval::{ApprovalPolicy, ExecApprovalRequest, ReviewDecision};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::execpolicy::{Decision, ExecPolicy};
use crate::sandbox::SandboxPolicy;
use std::sync::{Arc, Mutex};

pub fn tokenize_shell_command(command: &str) -> Vec<String> {
    shlex::split(command)
        .unwrap_or_else(|| command.split_whitespace().map(str::to_string).collect())
}

/// Returns effective sandbox policy after policy + approval checks.
pub fn ensure_shell_allowed(
    ctx: &ToolCallContext,
    args: &Value,
    command: &str,
) -> Result<SandboxPolicy, ToolError> {
    let tokens = tokenize_shell_command(command);
    if tokens.is_empty() {
        return Err(ToolError::respond("empty command"));
    }

    let cache = &ctx.approval_cache;
    if cache.is_prefix_allowed(&tokens) {
        return Ok(effective_sandbox_policy(ctx, args));
    }

    let policy_match = ctx.exec_policy.lock().unwrap().evaluate(&tokens);

    if policy_match.decision == Decision::Forbidden {
        let msg = policy_match
            .justification
            .unwrap_or_else(|| "command forbidden by execpolicy".into());
        return Err(ToolError::respond(format!("execpolicy forbidden: {msg}")));
    }

    let sandbox_perm = args
        .get("sandbox_permissions")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let needs_escalation = sandbox_perm.as_deref() == Some("require_escalated");
    let needs_prompt = policy_match.decision == Decision::Prompt
        || needs_escalation
        || cache.strict_auto_review()
        || args.get("justification").and_then(|v| v.as_str()).is_some();

    if !needs_prompt {
        return Ok(effective_sandbox_policy(ctx, args));
    }

    let approval_policy = ctx.session.approval_policy();
    match approval_policy {
        ApprovalPolicy::Always => return Ok(effective_sandbox_policy(ctx, args)),
        ApprovalPolicy::Never => {
            return Err(ToolError::respond(
                "command requires approval but approval_policy=never",
            ));
        }
        ApprovalPolicy::OnRequest => {}
    }

    let reason = args
        .get("justification")
        .or_else(|| args.get("reason"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or(policy_match.justification.clone())
        .or_else(|| {
            if needs_escalation {
                Some("sandbox_permissions=require_escalated".into())
            } else {
                None
            }
        });

    let req = ExecApprovalRequest {
        command: tokens.clone(),
        cwd: ctx.cwd.clone(),
        reason,
        sandbox_permissions: sandbox_perm,
        policy_decision: policy_match.decision,
        justification: policy_match.justification.clone(),
        proposed_execpolicy_amendment: Some(tokens.clone()),
    };

    let decision = ctx.session.request_exec_approval(req)?;
    apply_review_decision(ctx, &decision, &tokens)?;

    if decision.is_approved() {
        // A plain `Approved` decision covers this single execution only — it
        // is intentionally not cached, so an identical command still goes
        // through the same policy/approval checks next time (Codex parity;
        // use `ApprovedForSession` or an execpolicy amendment to persist).
        Ok(effective_sandbox_policy(ctx, args))
    } else {
        Err(ToolError::respond(format!(
            "exec approval denied: {:?}",
            decision
        )))
    }
}

fn apply_review_decision(
    ctx: &ToolCallContext,
    decision: &ReviewDecision,
    tokens: &[String],
) -> Result<(), ToolError> {
    match decision {
        ReviewDecision::Approved => Ok(()),
        ReviewDecision::ApprovedForSession => {
            ctx.approval_cache.allow_prefix_for_session(tokens.to_vec());
            Ok(())
        }
        ReviewDecision::ApprovedExecpolicyAmendment {
            proposed_execpolicy_amendment,
        } => {
            ctx.exec_policy
                .lock()
                .unwrap()
                .add_allow_prefix(proposed_execpolicy_amendment.clone());
            Ok(())
        }
        ReviewDecision::Denied | ReviewDecision::Abort => Ok(()),
    }
}

pub fn args_dangerously_disable_sandbox(ctx: &ToolCallContext, args: &Value) -> bool {
    args.get("dangerously_disable_sandbox")
        .and_then(|v| v.as_bool())
        .unwrap_or(ctx.dangerously_disable_sandbox)
}

pub fn effective_sandbox_policy(ctx: &ToolCallContext, args: &Value) -> SandboxPolicy {
    if args_dangerously_disable_sandbox(ctx, args) {
        return SandboxPolicy::Off;
    }
    match args.get("sandbox_permissions").and_then(|v| v.as_str()) {
        Some("require_escalated") => SandboxPolicy::Off,
        Some("with_additional_permissions") => {
            if ctx.approval_cache.effective_grants().is_effectively_empty() {
                ctx.sandbox_policy
            } else {
                SandboxPolicy::Restricted
            }
        }
        _ => ctx.sandbox_policy,
    }
}

pub fn new_exec_policy() -> Arc<Mutex<ExecPolicy>> {
    Arc::new(Mutex::new(ExecPolicy::default_codex_like()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::ApprovalCache;
    use crate::context::{LocalToolSession, ToolCallContext};
    use crate::discovery::new_tool_catalog;
    use crate::unified_exec::UnifiedExecManager;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn strict_ctx(
        exec_approval_calls: Arc<AtomicUsize>,
        decision: ReviewDecision,
    ) -> ToolCallContext {
        let session = Arc::new(LocalToolSession::default().with_exec_approval(move |_req| {
            exec_approval_calls.fetch_add(1, Ordering::SeqCst);
            Ok(decision.clone())
        }));
        let ctx = ToolCallContext::new(
            ".",
            "off",
            session,
            Arc::new(UnifiedExecManager::new()),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            new_tool_catalog(),
        );
        // Force `needs_prompt` regardless of execpolicy defaults so every
        // call below exercises the approval path under review.
        ctx.approval_cache.set_strict_auto_review(true);
        ctx
    }

    /// Regression test: a plain `Approved` decision must not be cached, or
    /// `strict_auto_review`'s "review every subsequent command" guarantee
    /// would be silently defeated after the first approval.
    #[test]
    fn approved_decision_is_not_cached_across_calls() {
        let calls = Arc::new(AtomicUsize::new(0));
        let ctx = strict_ctx(calls.clone(), ReviewDecision::Approved);

        let args = serde_json::json!({});
        ensure_shell_allowed(&ctx, &args, "echo hi").expect("first call approved");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        ensure_shell_allowed(&ctx, &args, "echo hi").expect("second call approved");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "identical command must be re-reviewed, not auto-approved from a stale cache"
        );
    }

    /// `ApprovedForSession` is the documented persistent-approval path and
    /// must keep working: it should still skip re-prompting for the
    /// remainder of the session (unlike the one-time `Approved` case above).
    #[test]
    fn approved_for_session_still_skips_future_prompts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let ctx = strict_ctx(calls.clone(), ReviewDecision::ApprovedForSession);

        let args = serde_json::json!({});
        ensure_shell_allowed(&ctx, &args, "echo hi").expect("first call approved");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        ensure_shell_allowed(&ctx, &args, "echo hi").expect("second call approved");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "ApprovedForSession must not re-prompt for the same command"
        );
    }

    #[test]
    fn effective_policy_maps_sandbox_permissions() {
        let cache = Arc::new(ApprovalCache::default());
        let ctx = ToolCallContext::new(
            ".",
            "restricted",
            Arc::new(LocalToolSession::default()),
            Arc::new(UnifiedExecManager::new()),
            new_exec_policy(),
            cache.clone(),
            new_tool_catalog(),
        );
        let base = serde_json::json!({});
        assert_eq!(
            effective_sandbox_policy(&ctx, &base),
            SandboxPolicy::Restricted
        );
        assert_eq!(
            effective_sandbox_policy(
                &ctx,
                &serde_json::json!({"sandbox_permissions": "require_escalated"}),
            ),
            SandboxPolicy::Off
        );
        assert_eq!(
            effective_sandbox_policy(
                &ctx,
                &serde_json::json!({"sandbox_permissions": "with_additional_permissions"}),
            ),
            SandboxPolicy::Restricted,
            "without grants, with_additional_permissions falls back to base policy"
        );
        cache.record_permission_grant(
            crate::approval::RequestPermissionProfile {
                network: Some(serde_json::json!({"enabled": true})),
                file_system: None,
            },
            "turn",
        );
        let ctx_off = ToolCallContext::new(
            ".",
            "off",
            Arc::new(LocalToolSession::default()),
            Arc::new(UnifiedExecManager::new()),
            new_exec_policy(),
            cache,
            new_tool_catalog(),
        );
        assert_eq!(
            effective_sandbox_policy(
                &ctx_off,
                &serde_json::json!({"sandbox_permissions": "with_additional_permissions"}),
            ),
            SandboxPolicy::Restricted,
            "granted permissions enable restricted overlay on off base policy"
        );
        assert_eq!(
            effective_sandbox_policy(
                &ctx,
                &serde_json::json!({"dangerously_disable_sandbox": true}),
            ),
            SandboxPolicy::Off
        );
    }

    /// Denied decisions must never be cached as approved.
    #[test]
    fn denied_decision_keeps_failing_and_reprompts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let ctx = strict_ctx(calls.clone(), ReviewDecision::Denied);

        let args = serde_json::json!({});
        assert!(ensure_shell_allowed(&ctx, &args, "echo hi").is_err());
        assert!(ensure_shell_allowed(&ctx, &args, "echo hi").is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
