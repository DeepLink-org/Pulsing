use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::approval::{ApprovalCache, new_exec_policy};
use crate::context::{LocalToolSession, NullToolSession, ToolCallContext, ToolSession};
use crate::discovery::{ToolCatalog, new_tool_catalog};
use crate::executor::ToolExecutor;
use crate::handlers::builtin_handlers;
use crate::handlers::try_call_mcp_dynamic_tool;
use crate::result::ToolResult;
use crate::unified_exec::UnifiedExecManager;

pub struct ToolRuntimeConfig {
    pub cwd: std::path::PathBuf,
    pub sandbox_policy: String,
    pub dangerously_disable_sandbox: bool,
    pub session: Arc<dyn ToolSession>,
    pub exec: Arc<UnifiedExecManager>,
    pub exec_policy: Arc<std::sync::Mutex<crate::execpolicy::ExecPolicy>>,
    pub approval_cache: Arc<ApprovalCache>,
    pub tool_catalog: Arc<std::sync::Mutex<ToolCatalog>>,
    pub mcp_runtime: Option<crate::mcp::SharedMcpRuntime>,
}

impl Default for ToolRuntimeConfig {
    fn default() -> Self {
        let catalog = new_tool_catalog();
        {
            let mut c = catalog.lock().unwrap();
            c.load_codex_plugins(&[]);
        }
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            sandbox_policy: "off".into(),
            dangerously_disable_sandbox: false,
            session: Arc::new(NullToolSession),
            exec: Arc::new(UnifiedExecManager::new()),
            exec_policy: new_exec_policy(),
            approval_cache: Arc::new(ApprovalCache::default()),
            tool_catalog: catalog,
            mcp_runtime: None,
        }
    }
}

pub struct ToolRuntime {
    handlers: HashMap<String, Box<dyn ToolExecutor>>,
    context: ToolCallContext,
}

impl ToolRuntime {
    pub fn new(config: ToolRuntimeConfig) -> Self {
        let mut handlers = HashMap::new();
        for h in builtin_handlers() {
            handlers.insert(h.tool_name().to_string(), h);
        }
        let context = ToolCallContext::new(
            config.cwd,
            &config.sandbox_policy,
            config.session,
            config.exec,
            config.exec_policy,
            config.approval_cache,
            config.tool_catalog,
        )
        .with_dangerous_sandbox(config.dangerously_disable_sandbox);
        let context = if let Some(mcp) = config.mcp_runtime {
            context.with_mcp_runtime(mcp)
        } else {
            context
        };
        Self { handlers, context }
    }

    pub fn with_local_session(cwd: impl AsRef<Path>, sandbox_policy: &str) -> Self {
        Self::new(ToolRuntimeConfig {
            cwd: cwd.as_ref().to_path_buf(),
            sandbox_policy: sandbox_policy.to_string(),
            session: Arc::new(LocalToolSession::default()),
            ..Default::default()
        })
    }

    pub fn context(&self) -> &ToolCallContext {
        &self.context
    }

    pub fn tool_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.handlers.keys().cloned().collect();
        names.sort();
        names
    }

    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> ToolResult {
        if let Some(h) = self.handlers.get(name) {
            return match h.handle(&self.context, arguments).await {
                Ok(r) => r,
                Err(e) => ToolResult::err(e.to_string()),
            };
        }
        if let Some(result) = try_call_mcp_dynamic_tool(&self.context, name, arguments).await {
            return match result {
                Ok(r) => r,
                Err(e) => ToolResult::err(e.to_string()),
            };
        }
        ToolResult::err(format!("Unknown tool: {name}"))
    }
}

impl Default for ToolRuntime {
    fn default() -> Self {
        Self::with_local_session(".", "off")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{PlanItem, StepStatus};

    #[tokio::test]
    async fn mcp_dynamic_tool_routes_before_unknown() {
        use rmcp::model::Tool;
        use std::sync::Arc;

        let tool = Tool::new_with_raw(
            "echo".to_string(),
            Some("Echo".into()),
            Arc::new(serde_json::Map::new()),
        );
        let info = crate::mcp::ToolInfo {
            server_name: "demo".into(),
            supports_parallel_tool_calls: false,
            server_origin: None,
            callable_name: "echo".into(),
            callable_namespace: "demo".into(),
            namespace_description: None,
            tool,
            connector_id: None,
            connector_name: None,
            plugin_display_names: vec![],
        };
        let catalog = crate::discovery::new_tool_catalog();
        let catalog_snap =
            crate::mcp::build_default_catalog(vec![], std::collections::HashMap::new());
        let manager = crate::mcp::McpConnectionManager::from_tools(vec![info], true);
        let mcp = crate::mcp::McpRuntime {
            catalog: catalog_snap,
            manager,
        };
        let slot = crate::mcp::new_shared_mcp_runtime();
        *slot.write().await = Some(mcp);
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            mcp_runtime: Some(slot),
            tool_catalog: catalog,
            ..Default::default()
        });
        let out = rt.call_tool("mcp__demo__echo", serde_json::json!({})).await;
        assert!(
            !out.content.starts_with("Unknown tool:"),
            "expected MCP dispatch, got: {}",
            out.content
        );
        assert!(out.is_error);
        assert!(
            out.content.contains("MCP server not connected") || out.content.contains("MCP tool")
        );
    }

    #[tokio::test]
    async fn read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let fp = dir.path().join("a.txt");
        std::fs::write(&fp, "hello").unwrap();
        let rt = ToolRuntime::with_local_session(dir.path(), "off");
        let out = rt
            .call_tool("Read", serde_json::json!({"file_path": "a.txt"}))
            .await;
        assert!(!out.is_error);
        assert_eq!(out.content, "hello");
    }

    #[tokio::test]
    async fn update_plan_uses_session() {
        let session = Arc::new(LocalToolSession::default());
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            session: session.clone(),
            ..Default::default()
        });
        let out = rt
            .call_tool(
                "update_plan",
                serde_json::json!({
                    "plan": [{"step": "one", "status": "pending"}]
                }),
            )
            .await;
        assert!(!out.is_error);
        assert_eq!(
            session.plan_snapshot(),
            vec![PlanItem {
                step: "one".into(),
                status: StepStatus::Pending,
            }]
        );
    }

    #[tokio::test]
    async fn shell_command_codex_args() {
        let session = Arc::new(
            LocalToolSession::default()
                .with_approval_policy(crate::approval::ApprovalPolicy::Always),
        );
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            session,
            ..Default::default()
        });
        let out = rt
            .call_tool(
                "shell_command",
                serde_json::json!({"cmd": "echo hi", "workdir": "."}),
            )
            .await;
        assert!(!out.is_error);
        assert!(out.content.contains("hi"));
    }

    #[tokio::test]
    async fn bash_alias_matches_shell_command() {
        let session = Arc::new(
            LocalToolSession::default()
                .with_approval_policy(crate::approval::ApprovalPolicy::Always),
        );
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            session,
            ..Default::default()
        });
        let out = rt
            .call_tool("Bash", serde_json::json!({"command": "echo alias-ok"}))
            .await;
        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.contains("alias-ok"));
    }

    #[tokio::test]
    async fn shell_command_login_does_not_bypass_restricted_sandbox() {
        let session = Arc::new(
            LocalToolSession::default()
                .with_approval_policy(crate::approval::ApprovalPolicy::Always),
        );
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            session,
            sandbox_policy: "restricted".into(),
            ..Default::default()
        });
        let out = rt
            .call_tool(
                "shell_command",
                serde_json::json!({
                    "cmd": "echo hi",
                    "login": true,
                }),
            )
            .await;
        assert!(!out.is_error, "{}", out.content);
        assert!(
            out.content.contains("restricted env"),
            "login shell must stay inside restricted wrapper, got: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn shell_command_require_escalated_denied_when_approval_never() {
        let session = Arc::new(
            LocalToolSession::default()
                .with_approval_policy(crate::approval::ApprovalPolicy::Never),
        );
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            session,
            sandbox_policy: "restricted".into(),
            ..Default::default()
        });
        let out = rt
            .call_tool(
                "shell_command",
                serde_json::json!({
                    "cmd": "echo hi",
                    "sandbox_permissions": "require_escalated",
                }),
            )
            .await;
        assert!(
            out.is_error,
            "require_escalated must be denied when approval_policy=never"
        );
    }

    #[tokio::test]
    async fn tool_search_finds_deferred_tool() {
        let catalog = crate::discovery::new_tool_catalog();
        {
            let mut c = catalog.lock().unwrap();
            c.register_deferred(crate::discovery::DeferredToolEntry::from_function(
                "github_mcp",
                "GitHub MCP integration",
                serde_json::json!({"type": "object"}),
            ));
        }
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            tool_catalog: catalog,
            ..Default::default()
        });
        let out = rt
            .call_tool("tool_search", serde_json::json!({"query": "github mcp"}))
            .await;
        assert!(!out.is_error);
        assert!(out.content.contains("github_mcp"));
    }

    #[tokio::test]
    async fn tool_search_rejects_empty_query() {
        let rt = ToolRuntime::new(ToolRuntimeConfig::default());
        for args in [
            serde_json::json!({"query": ""}),
            serde_json::json!({"query": "   "}),
            serde_json::json!({}),
        ] {
            let out = rt.call_tool("tool_search", args).await;
            assert!(out.is_error);
        }
    }

    #[tokio::test]
    async fn tool_search_limit_zero_or_negative_falls_back_to_default() {
        let catalog = crate::discovery::new_tool_catalog();
        {
            let mut c = catalog.lock().unwrap();
            for name in ["github_mcp", "github_issues", "github_actions"] {
                c.register_deferred(crate::discovery::DeferredToolEntry::from_function(
                    name,
                    "GitHub integration",
                    serde_json::json!({"type": "object"}),
                ));
            }
        }
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            tool_catalog: catalog,
            ..Default::default()
        });
        // limit=0 and limit=-5 are not usable positive counts, so both should
        // behave like the omitted-limit (default) case rather than returning
        // zero or panicking.
        for args in [
            serde_json::json!({"query": "github", "limit": 0}),
            serde_json::json!({"query": "github", "limit": -5}),
        ] {
            let out = rt.call_tool("tool_search", args).await;
            assert!(!out.is_error);
            let payload: serde_json::Value = serde_json::from_str(&out.content).unwrap();
            assert_eq!(payload["tools"].as_array().unwrap().len(), 3);
        }
    }

    #[tokio::test]
    async fn tool_search_huge_limit_is_clamped_not_rejected() {
        let catalog = crate::discovery::new_tool_catalog();
        {
            let mut c = catalog.lock().unwrap();
            c.register_deferred(crate::discovery::DeferredToolEntry::from_function(
                "github_mcp",
                "GitHub MCP integration",
                serde_json::json!({"type": "object"}),
            ));
        }
        let rt = ToolRuntime::new(ToolRuntimeConfig {
            tool_catalog: catalog,
            ..Default::default()
        });
        let out = rt
            .call_tool(
                "tool_search",
                serde_json::json!({"query": "github", "limit": 999_999_999_u64}),
            )
            .await;
        assert!(!out.is_error);
        assert!(out.content.contains("github_mcp"));
    }
}
