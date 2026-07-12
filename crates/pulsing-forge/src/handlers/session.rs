use serde_json::Value;

use super::ok;
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

use crate::session_input::{args_to_value, validate_request_user_input};

/// Must stay byte-for-byte identical to `NEW_CONTEXT_MESSAGE` in
/// `python/pulsing/forge/handlers.py` — that pure-Python fallback path does not
/// link against this crate, so the string is duplicated rather than shared.
pub const NEW_CONTEXT_MESSAGE: &str =
    "A new context window will start without summarizing conversation history.";

pub struct NewContextHandler;

impl ToolExecutor for NewContextHandler {
    fn tool_name(&self) -> &str {
        "new_context"
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, _arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { new_context_impl(ctx) })
    }
}

fn new_context_impl(ctx: &ToolCallContext) -> Result<crate::result::ToolResult, ToolError> {
    ctx.approval_cache.clear_turn_state();
    ctx.session.request_new_context()?;
    ok(NEW_CONTEXT_MESSAGE)
}

pub struct GetContextRemainingHandler;

impl ToolExecutor for GetContextRemainingHandler {
    fn tool_name(&self) -> &str {
        "get_context_remaining"
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, _arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { get_context_remaining_impl(ctx) })
    }
}

fn get_context_remaining_impl(
    ctx: &ToolCallContext,
) -> Result<crate::result::ToolResult, ToolError> {
    let remaining = ctx.session.tokens_remaining();
    let payload = match remaining {
        Some(n) => serde_json::json!({
            "tokens_remaining": n,
            "status": "ok",
        }),
        None => serde_json::json!({
            "tokens_remaining": null,
            "status": "unknown",
        }),
    };
    let text = serde_json::to_string_pretty(&payload)
        .map_err(|e| ToolError::respond(format!("failed to encode context remaining: {e}")))?;
    ok(text)
}

pub struct RequestUserInputHandler;

impl ToolExecutor for RequestUserInputHandler {
    fn tool_name(&self) -> &str {
        "request_user_input"
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { request_user_input_impl(ctx, arguments) })
    }
}

fn request_user_input_impl(
    ctx: &ToolCallContext,
    arguments: Value,
) -> Result<crate::result::ToolResult, ToolError> {
    let validated = validate_request_user_input(&arguments)?;
    let response = ctx.session.request_user_input(args_to_value(&validated))?;
    let text = serde_json::to_string_pretty(&response)
        .map_err(|e| ToolError::respond(format!("failed to encode user input response: {e}")))?;
    ok(text)
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

    struct FailingSession;

    impl crate::context::ToolSession for FailingSession {
        fn update_plan(&self, _args: crate::context::UpdatePlanArgs) -> Result<(), ToolError> {
            Ok(())
        }

        fn request_new_context(&self) -> Result<(), ToolError> {
            Err(ToolError::respond("session unavailable"))
        }

        fn tokens_remaining(&self) -> Option<i64> {
            None
        }

        fn request_user_input(&self, _arguments: Value) -> Result<Value, ToolError> {
            Err(ToolError::respond("not supported"))
        }
    }

    #[test]
    fn new_context_requests_reset_and_returns_message() {
        let session = Arc::new(LocalToolSession::default());
        let ctx = test_ctx(session.clone());
        let out = new_context_impl(&ctx).unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, NEW_CONTEXT_MESSAGE);
        assert!(session.new_context_requested());
    }

    #[test]
    fn new_context_propagates_session_error() {
        let ctx = test_ctx(Arc::new(FailingSession));
        let err = new_context_impl(&ctx).unwrap_err();
        assert_eq!(err.to_string(), "session unavailable");
    }

    #[test]
    fn get_context_remaining_reports_budget_when_known() {
        let ctx = test_ctx(Arc::new(LocalToolSession::new(Some(42_000))));
        let out = get_context_remaining_impl(&ctx).unwrap();
        assert!(!out.is_error);
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        assert_eq!(payload["tokens_remaining"], 42_000);
        assert_eq!(payload["status"], "ok");
    }

    #[test]
    fn get_context_remaining_reports_unknown_without_budget() {
        let ctx = test_ctx(Arc::new(LocalToolSession::default()));
        let out = get_context_remaining_impl(&ctx).unwrap();
        assert!(!out.is_error);
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        assert!(payload["tokens_remaining"].is_null());
        assert_eq!(payload["status"], "unknown");
    }

    #[test]
    fn request_user_input_rejects_empty_questions() {
        let ctx = test_ctx(Arc::new(LocalToolSession::default()));
        let err =
            request_user_input_impl(&ctx, serde_json::json!({ "questions": [] })).unwrap_err();
        assert!(err.to_string().contains("question"));
    }

    #[test]
    fn request_user_input_reports_malformed_auto_resolution_ms() {
        let ctx = test_ctx(Arc::new(LocalToolSession::default()));
        let err = request_user_input_impl(
            &ctx,
            serde_json::json!({
                "questions": [{"id": "q1", "header": "H", "question": "Q?"}],
                "autoResolutionMs": {"bad": true}
            }),
        )
        .unwrap_err();
        assert!(err.to_string().to_lowercase().contains("autoresolutionms"));
    }

    #[test]
    fn request_user_input_passes_clamped_payload_to_session() {
        use crate::session_input::MIN_AUTO_RESOLUTION_MS;
        use std::sync::Mutex;

        let seen = Arc::new(Mutex::new(None::<Value>));
        let seen_cb = seen.clone();
        let session = Arc::new(LocalToolSession::default().with_user_input(move |args| {
            *seen_cb.lock().unwrap() = Some(args);
            Ok(serde_json::json!({
                "answers": { "q1": { "answers": ["A"] } }
            }))
        }));
        let ctx = test_ctx(session);
        let out = request_user_input_impl(
            &ctx,
            serde_json::json!({
                "questions": [{
                    "id": "q1",
                    "header": "Pick",
                    "question": "Which?",
                    "options": [{"label": "A", "description": "first"}]
                }],
                "autoResolutionMs": 1
            }),
        )
        .unwrap();
        assert!(!out.is_error);
        let payload = seen
            .lock()
            .unwrap()
            .clone()
            .expect("session callback invoked");
        assert_eq!(payload["autoResolutionMs"], MIN_AUTO_RESOLUTION_MS);
    }
}
