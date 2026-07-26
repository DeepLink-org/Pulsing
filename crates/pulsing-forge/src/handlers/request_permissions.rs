use serde_json::Value;

use super::ok;
use crate::approval::RequestPermissionsArgs;
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

pub struct RequestPermissionsHandler;

impl ToolExecutor for RequestPermissionsHandler {
    fn tool_name(&self) -> &str {
        "request_permissions"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { request_permissions_impl(ctx, arguments) })
    }
}

fn request_permissions_impl(
    ctx: &ToolCallContext,
    arguments: Value,
) -> Result<crate::result::ToolResult, ToolError> {
    let args: RequestPermissionsArgs = serde_json::from_value(arguments)
        .map_err(|e| ToolError::respond(format!("failed to parse request_permissions: {e}")))?;
    if !args.permissions.has_actionable_request() {
        return Err(ToolError::respond(
            "request_permissions requires non-empty network or file_system permissions",
        ));
    }
    args.permissions
        .validate_paths(&ctx.cwd)
        .map_err(ToolError::respond)?;
    let response = ctx.session.request_permissions(args)?;
    if response.permissions.is_effectively_empty() {
        return Err(ToolError::respond("permissions denied by host"));
    }
    ctx.approval_cache
        .record_permission_grant(response.permissions.clone(), &response.scope);
    if response.strict_auto_review {
        ctx.approval_cache.set_strict_auto_review(true);
    }
    let text = serde_json::to_string_pretty(&response)
        .map_err(|e| ToolError::respond(format!("encode request_permissions response: {e}")))?;
    ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::{
        ApprovalCache, RequestPermissionProfile, RequestPermissionsArgs, RequestPermissionsResponse,
    };
    use crate::context::LocalToolSession;
    use crate::runtime::{ToolRuntime, ToolRuntimeConfig};
    use std::sync::Arc;

    fn rt_with_perms<F>(f: F) -> ToolRuntime
    where
        F: Fn(RequestPermissionsArgs) -> Result<RequestPermissionsResponse, ToolError>
            + Send
            + Sync
            + 'static,
    {
        let session = Arc::new(LocalToolSession::default().with_request_permissions(f));
        ToolRuntime::new(ToolRuntimeConfig {
            session,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn rejects_missing_permission_sections() {
        let rt = rt_with_perms(|_| {
            Ok(RequestPermissionsResponse {
                permissions: RequestPermissionProfile {
                    network: Some(serde_json::json!({"enabled": true})),
                    file_system: None,
                },
                scope: "turn".into(),
                strict_auto_review: false,
            })
        });
        let out = rt
            .call_tool(
                "request_permissions",
                serde_json::json!({"permissions": {}}),
            )
            .await;
        assert!(out.is_error);
        assert!(out.content.contains("non-empty"));
    }

    #[tokio::test]
    async fn rejects_empty_nested_network_object() {
        let rt = rt_with_perms(|_| {
            Ok(RequestPermissionsResponse {
                permissions: RequestPermissionProfile {
                    network: Some(serde_json::json!({"enabled": true})),
                    file_system: None,
                },
                scope: "turn".into(),
                strict_auto_review: false,
            })
        });
        let out = rt
            .call_tool(
                "request_permissions",
                serde_json::json!({"permissions": {"network": {}}}),
            )
            .await;
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn rejects_host_empty_grant() {
        let rt = rt_with_perms(|_| {
            Ok(RequestPermissionsResponse {
                permissions: RequestPermissionProfile::default(),
                scope: "turn".into(),
                strict_auto_review: false,
            })
        });
        let out = rt
            .call_tool(
                "request_permissions",
                serde_json::json!({"permissions": {"network": {"enabled": true}}}),
            )
            .await;
        assert!(out.is_error);
        assert!(out.content.contains("denied"));
    }

    #[tokio::test]
    async fn sets_strict_auto_review_on_host_grant() {
        let cache = Arc::new(ApprovalCache::default());
        let session = Arc::new(
            LocalToolSession::default().with_request_permissions(|args| {
                Ok(RequestPermissionsResponse {
                    permissions: args.permissions,
                    scope: "turn".into(),
                    strict_auto_review: true,
                })
            }),
        );
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            session,
            approval_cache: cache.clone(),
            ..Default::default()
        });
        let out = rt
            .call_tool(
                "request_permissions",
                serde_json::json!({"permissions": {"network": {"enabled": true}}}),
            )
            .await;
        assert!(!out.is_error);
        assert!(cache.strict_auto_review());
    }

    #[tokio::test]
    async fn new_context_clears_strict_auto_review() {
        let cache = Arc::new(ApprovalCache::default());
        cache.set_strict_auto_review(true);
        cache.record_permission_grant(
            RequestPermissionProfile {
                network: Some(serde_json::json!({"enabled": true})),
                file_system: None,
            },
            "turn",
        );
        let session = Arc::new(LocalToolSession::default());
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            session,
            approval_cache: cache.clone(),
            ..Default::default()
        });
        let out = rt.call_tool("new_context", serde_json::json!({})).await;
        assert!(!out.is_error);
        assert!(!cache.strict_auto_review());
        assert!(!cache.network_granted());
    }

    #[tokio::test]
    async fn records_session_grant_after_approval() {
        let cache = Arc::new(ApprovalCache::default());
        let session = Arc::new(
            LocalToolSession::default().with_request_permissions(|args| {
                Ok(RequestPermissionsResponse {
                    permissions: args.permissions,
                    scope: "session".into(),
                    strict_auto_review: false,
                })
            }),
        );
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            session,
            approval_cache: cache.clone(),
            ..Default::default()
        });
        let out = rt
            .call_tool(
                "request_permissions",
                serde_json::json!({"permissions": {"network": {"enabled": true}}}),
            )
            .await;
        assert!(!out.is_error);
        assert!(cache.network_granted());
    }

    #[tokio::test]
    async fn rejects_path_outside_cwd() {
        let rt = rt_with_perms(|_| {
            Ok(RequestPermissionsResponse {
                permissions: RequestPermissionProfile {
                    network: None,
                    file_system: Some(serde_json::json!({"write": ["/etc/passwd"]})),
                },
                scope: "turn".into(),
                strict_auto_review: false,
            })
        });
        let out = rt
            .call_tool(
                "request_permissions",
                serde_json::json!({"permissions": {"file_system": {"write": ["/etc/passwd"]}}}),
            )
            .await;
        assert!(out.is_error);
        assert!(out.content.contains("outside working directory"));
    }

    #[test]
    fn permission_profile_detects_codex_empty_shapes() {
        let profile = RequestPermissionProfile {
            network: Some(serde_json::json!({"enabled": null})),
            file_system: Some(serde_json::json!({"entries": []})),
        };
        assert!(profile.is_effectively_empty());
        assert!(!profile.has_actionable_request());
    }
}
