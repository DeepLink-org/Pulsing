use serde_json::Value;

use super::ok;
use crate::context::{StepStatus, ToolCallContext, UpdatePlanArgs};
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

/// Must stay byte-for-byte identical to `PLAN_UPDATED` in
/// `python/pulsing/forge/handlers.py` — that pure-Python fallback path does not
/// link against this crate, so the string is duplicated rather than shared.
const PLAN_UPDATED: &str = "Plan updated";

pub struct UpdatePlanHandler;

impl ToolExecutor for UpdatePlanHandler {
    fn tool_name(&self) -> &str {
        "update_plan"
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { update_plan_impl(ctx, arguments) })
    }
}

fn validate_update_plan(value: &Value) -> Result<UpdatePlanArgs, ToolError> {
    let args: UpdatePlanArgs = serde_json::from_value(value.clone())
        .map_err(|e| ToolError::respond(format!("failed to parse update_plan arguments: {e}")))?;

    let in_progress = args
        .plan
        .iter()
        .filter(|item| item.status == StepStatus::InProgress)
        .count();
    if in_progress > 1 {
        return Err(ToolError::respond(
            "update_plan allows at most one step with status \"in_progress\"",
        ));
    }

    Ok(args)
}

fn update_plan_impl(
    ctx: &ToolCallContext,
    arguments: Value,
) -> Result<crate::result::ToolResult, ToolError> {
    let args = validate_update_plan(&arguments)?;
    ctx.session.update_plan(args)?;
    ok(PLAN_UPDATED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::{ApprovalCache, new_exec_policy};
    use crate::context::LocalToolSession;
    use crate::discovery::new_tool_catalog;
    use crate::unified_exec::UnifiedExecManager;
    use std::sync::Arc;

    fn test_ctx(session: Arc<dyn crate::context::ToolSession>) -> ToolCallContext {
        ToolCallContext::new(
            ".",
            "off",
            session,
            Arc::new(UnifiedExecManager::new()),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            new_tool_catalog(),
        )
    }

    #[test]
    fn rejects_multiple_in_progress() {
        let raw = serde_json::json!({
            "plan": [
                {"step": "one", "status": "in_progress"},
                {"step": "two", "status": "in_progress"}
            ]
        });
        let err = validate_update_plan(&raw).unwrap_err();
        assert_eq!(
            err,
            ToolError::respond("update_plan allows at most one step with status \"in_progress\"")
        );
    }

    #[test]
    fn accepts_single_in_progress() {
        let raw = serde_json::json!({
            "plan": [
                {"step": "one", "status": "completed"},
                {"step": "two", "status": "in_progress"}
            ]
        });
        let args = validate_update_plan(&raw).unwrap();
        assert_eq!(args.plan.len(), 2);
    }

    #[test]
    fn rejects_missing_plan_field() {
        let raw = serde_json::json!({});
        let err = validate_update_plan(&raw).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("plan"),
            "expected parse error mentioning plan, got: {msg}"
        );
    }

    #[test]
    fn update_plan_impl_updates_session() {
        let session = Arc::new(LocalToolSession::default());
        let ctx = test_ctx(session.clone());
        let out = update_plan_impl(
            &ctx,
            serde_json::json!({
                "plan": [{"step": "one", "status": "pending"}]
            }),
        )
        .unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, PLAN_UPDATED);
        assert_eq!(session.plan_snapshot().len(), 1);
    }
}
