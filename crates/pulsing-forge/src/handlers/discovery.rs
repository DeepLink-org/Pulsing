use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ok;
use crate::context::ToolCallContext;
use crate::discovery::{
    DeferredToolEntry, DiscoverablePlugin, TOOL_SEARCH_DEFAULT_LIMIT, TOOL_SEARCH_MAX_LIMIT,
};
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiscoverableToolType {
    Connector,
    Plugin,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiscoverableToolAction {
    Install,
    Enable,
}

#[derive(Debug, Deserialize)]
pub struct RequestPluginInstallArgs {
    #[serde(alias = "plugin_id", alias = "tool_id")]
    pub tool_id: String,
    #[serde(default)]
    pub tool_type: Option<DiscoverableToolType>,
    #[serde(default)]
    pub action_type: Option<DiscoverableToolAction>,
    #[serde(default, alias = "reason")]
    pub suggest_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RequestPluginInstallResult {
    pub completed: bool,
    pub user_confirmed: bool,
    pub tool_type: DiscoverableToolType,
    pub action_type: DiscoverableToolAction,
    pub tool_id: String,
    pub tool_name: String,
    pub suggest_reason: String,
    pub tools_registered: usize,
}

fn loadable_tool_json(entry: &DeferredToolEntry) -> Value {
    serde_json::json!({
        "type": "function",
        "name": entry.name,
        "description": entry.description,
        "parameters": entry.parameters,
        "defer_loading": entry.defer_loading,
        "namespace": entry.namespace,
        "plugin_id": entry.plugin_id,
        "source": entry.source,
    })
}

const DESCRIPTION_MAX_LEN: usize = 240;

fn truncate_description(desc: Option<&str>) -> Option<String> {
    desc.map(|s| {
        let char_count = s.chars().count();
        if char_count <= DESCRIPTION_MAX_LEN {
            s.to_string()
        } else {
            let truncated: String = s.chars().take(DESCRIPTION_MAX_LEN - 1).collect();
            format!("{truncated}…")
        }
    })
}

fn discoverable_entry_json(p: &DiscoverablePlugin) -> Value {
    serde_json::json!({
        "id": p.id,
        "name": p.name,
        "description": truncate_description(p.description.as_deref()),
        "tool_type": "plugin",
        "has_skills": p.has_skills,
        "mcp_server_names": p.mcp_server_names,
        "app_connector_ids": p.app_connector_ids,
    })
}

pub struct ToolSearchHandler;

impl ToolExecutor for ToolSearchHandler {
    fn tool_name(&self) -> &str {
        "tool_search"
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { tool_search_impl(ctx, arguments) })
    }
}

/// Defensive cap on query length so a pathological/huge query can't blow up
/// tokenization cost; Codex's tool_search runs server-side and has no such
/// concern, but this handler scores locally against the deferred-tool catalog.
const TOOL_SEARCH_MAX_QUERY_CHARS: usize = 2000;

fn tool_search_impl(
    ctx: &ToolCallContext,
    arguments: Value,
) -> Result<crate::result::ToolResult, ToolError> {
    let query = arguments
        .get("query")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ToolError::respond("tool_search requires non-empty query"))?;
    let query: String = query.chars().take(TOOL_SEARCH_MAX_QUERY_CHARS).collect();
    // `limit` must be a positive JSON integer; anything else (missing, 0,
    // negative, non-numeric) falls back to the default, and the result is
    // clamped so a huge value can't force an unbounded response.
    let limit = arguments
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .filter(|&n| n > 0)
        .unwrap_or(TOOL_SEARCH_DEFAULT_LIMIT)
        .min(TOOL_SEARCH_MAX_LIMIT);
    let hits = ctx.tool_catalog.lock().unwrap().search(&query, limit);
    let tools: Vec<Value> = hits.iter().map(loadable_tool_json).collect();
    let text = serde_json::to_string_pretty(&serde_json::json!({ "tools": tools }))
        .map_err(|e| ToolError::respond(e.to_string()))?;
    ok(text)
}

#[cfg(test)]
mod tool_search_tests {
    use super::*;
    use crate::approval::{ApprovalCache, new_exec_policy};
    use crate::context::LocalToolSession;
    use crate::discovery::{DeferredToolEntry, new_tool_catalog};
    use crate::unified_exec::UnifiedExecManager;
    use std::sync::Arc;

    fn test_ctx(catalog: Arc<std::sync::Mutex<crate::discovery::ToolCatalog>>) -> ToolCallContext {
        ToolCallContext::new(
            ".",
            "off",
            Arc::new(LocalToolSession::default()),
            Arc::new(UnifiedExecManager::new()),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            catalog,
        )
    }

    fn catalog_with_github_tool() -> Arc<std::sync::Mutex<crate::discovery::ToolCatalog>> {
        let catalog = new_tool_catalog();
        catalog
            .lock()
            .unwrap()
            .register_deferred(DeferredToolEntry::from_function(
                "github_mcp",
                "GitHub MCP integration",
                serde_json::json!({"type": "object"}),
            ));
        catalog
    }

    #[test]
    fn rejects_empty_query() {
        let ctx = test_ctx(new_tool_catalog());
        for args in [
            serde_json::json!({"query": ""}),
            serde_json::json!({"query": "   "}),
            serde_json::json!({}),
        ] {
            let err = tool_search_impl(&ctx, args).unwrap_err();
            assert!(err.to_string().contains("non-empty query"));
        }
    }

    #[test]
    fn returns_loadable_tool_json() {
        let ctx = test_ctx(catalog_with_github_tool());
        let out = tool_search_impl(&ctx, serde_json::json!({"query": "github mcp"})).unwrap();
        assert!(!out.is_error);
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        let tool = &payload["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "github_mcp");
        assert_eq!(tool["defer_loading"], true);
    }

    #[test]
    fn limit_zero_or_negative_falls_back_to_default() {
        let catalog = new_tool_catalog();
        for name in ["github_mcp", "github_issues", "github_actions"] {
            catalog
                .lock()
                .unwrap()
                .register_deferred(DeferredToolEntry::from_function(
                    name,
                    "GitHub integration",
                    serde_json::json!({"type": "object"}),
                ));
        }
        let ctx = test_ctx(catalog);
        for args in [
            serde_json::json!({"query": "github", "limit": 0}),
            serde_json::json!({"query": "github", "limit": -5}),
        ] {
            let out = tool_search_impl(&ctx, args).unwrap();
            let payload: Value = serde_json::from_str(&out.content).unwrap();
            assert_eq!(payload["tools"].as_array().unwrap().len(), 3);
        }
    }

    #[test]
    fn huge_limit_is_clamped() {
        let ctx = test_ctx(catalog_with_github_tool());
        let out = tool_search_impl(
            &ctx,
            serde_json::json!({"query": "github", "limit": 999_999_999_u64}),
        )
        .unwrap();
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        assert_eq!(payload["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn truncates_overlong_query() {
        let ctx = test_ctx(catalog_with_github_tool());
        let query = format!("github {}", "x".repeat(TOOL_SEARCH_MAX_QUERY_CHARS));
        let out = tool_search_impl(&ctx, serde_json::json!({"query": query})).unwrap();
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        assert_eq!(payload["tools"].as_array().unwrap().len(), 1);
    }
}

pub struct ListAvailablePluginsHandler;

impl ToolExecutor for ListAvailablePluginsHandler {
    fn tool_name(&self) -> &str {
        "list_available_plugins_to_install"
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, _arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { list_plugins_impl(ctx) })
    }
}

fn list_plugins_impl(ctx: &ToolCallContext) -> Result<crate::result::ToolResult, ToolError> {
    let plugins = {
        let mut catalog = ctx.tool_catalog.lock().unwrap();
        catalog
            .refresh_from_codex(&[])
            .map_err(ToolError::respond)?;
        catalog.list_installable()
    };
    let tools: Vec<Value> = plugins.iter().map(discoverable_entry_json).collect();
    let text = serde_json::to_string_pretty(&serde_json::json!({ "tools": tools }))
        .map_err(|e| ToolError::respond(e.to_string()))?;
    ok(text)
}

pub struct RequestPluginInstallHandler;

impl ToolExecutor for RequestPluginInstallHandler {
    fn tool_name(&self) -> &str {
        "request_plugin_install"
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        Box::pin(async move { request_plugin_install_impl(ctx, arguments) })
    }
}

fn request_plugin_install_impl(
    ctx: &ToolCallContext,
    arguments: Value,
) -> Result<crate::result::ToolResult, ToolError> {
    let args: RequestPluginInstallArgs = serde_json::from_value(arguments).map_err(|e| {
        ToolError::respond(format!("invalid request_plugin_install arguments: {e}"))
    })?;
    let tool_id = args.tool_id.trim().to_string();
    if tool_id.is_empty() {
        return Err(ToolError::respond(
            "request_plugin_install requires tool_id",
        ));
    }
    let tool_type = args.tool_type.unwrap_or(DiscoverableToolType::Plugin);
    let action_type = args.action_type.unwrap_or(DiscoverableToolAction::Install);
    if action_type != DiscoverableToolAction::Install {
        return Err(ToolError::respond(
            "plugin install requests currently support only action_type=\"install\"",
        ));
    }
    let suggest_reason = args
        .suggest_reason
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            ToolError::respond("request_plugin_install requires non-empty suggest_reason")
        })?;

    let plugin = ctx
        .tool_catalog
        .lock()
        .unwrap()
        .find_plugin(&tool_id)
        .ok_or_else(|| ToolError::respond(format!("unknown plugin {tool_id:?}")))?;

    let confirmed = ctx.session.request_plugin_install(
        tool_id.clone(),
        plugin.name.clone(),
        suggest_reason.clone(),
    )?;

    let tools_registered = if confirmed {
        ctx.tool_catalog
            .lock()
            .unwrap()
            .install_plugin(&tool_id)
            .map_err(ToolError::respond)?
            .len()
    } else {
        0
    };

    let result = RequestPluginInstallResult {
        completed: true,
        user_confirmed: confirmed,
        tool_type,
        action_type,
        tool_id,
        tool_name: plugin.name,
        suggest_reason,
        tools_registered,
    };
    let text =
        serde_json::to_string_pretty(&result).map_err(|e| ToolError::respond(e.to_string()))?;
    ok(text)
}

#[cfg(test)]
mod request_plugin_install_tests {
    use super::*;
    use crate::approval::{ApprovalCache, new_exec_policy};
    use crate::context::LocalToolSession;
    use crate::discovery::new_tool_catalog;
    use crate::unified_exec::UnifiedExecManager;
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    struct CatalogFixture {
        _dir: TempDir,
        catalog: Arc<std::sync::Mutex<crate::discovery::ToolCatalog>>,
        plugin_id: String,
    }

    fn test_ctx(
        session: Arc<dyn crate::context::ToolSession>,
        catalog: Arc<std::sync::Mutex<crate::discovery::ToolCatalog>>,
    ) -> ToolCallContext {
        ToolCallContext::new(
            ".",
            "off",
            session,
            Arc::new(UnifiedExecManager::new()),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            catalog,
        )
    }

    fn catalog_with_demo_plugin() -> CatalogFixture {
        let dir = TempDir::new().unwrap();
        let plugin_dir = dir.path().join("demo");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.json"),
            r#"{
                "id": "demo@local-dev",
                "name": "Demo Plugin",
                "description": "For tests",
                "tools": [{"name": "demo_tool", "description": "hello", "parameters": {"type":"object"}}]
            }"#,
        )
        .unwrap();
        let catalog = new_tool_catalog();
        catalog
            .lock()
            .unwrap()
            .load_codex_plugins(&[dir.path().to_path_buf()]);
        CatalogFixture {
            _dir: dir,
            catalog,
            plugin_id: "demo@local-dev".to_string(),
        }
    }

    #[test]
    fn request_plugin_install_requires_suggest_reason() {
        let ctx = test_ctx(
            Arc::new(LocalToolSession::default().with_plugin_install(|_, _, _| Ok(true))),
            new_tool_catalog(),
        );
        let err = request_plugin_install_impl(
            &ctx,
            serde_json::json!({
                "tool_id": "demo@local-dev",
                "suggest_reason": "   ",
            }),
        )
        .unwrap_err();
        assert!(err.to_string().contains("non-empty suggest_reason"));
    }

    #[test]
    fn request_plugin_install_rejects_enable_action() {
        let fixture = catalog_with_demo_plugin();
        let ctx = test_ctx(
            Arc::new(LocalToolSession::default().with_plugin_install(|_, _, _| Ok(true))),
            fixture.catalog,
        );
        let err = request_plugin_install_impl(
            &ctx,
            serde_json::json!({
                "tool_id": fixture.plugin_id,
                "action_type": "enable",
                "suggest_reason": "Need it",
            }),
        )
        .unwrap_err();
        assert!(err.to_string().contains("action_type=\"install\""));
    }

    #[test]
    fn request_plugin_install_unknown_plugin() {
        let ctx = test_ctx(
            Arc::new(LocalToolSession::default().with_plugin_install(|_, _, _| Ok(true))),
            new_tool_catalog(),
        );
        let err = request_plugin_install_impl(
            &ctx,
            serde_json::json!({
                "tool_id": "missing@local-dev",
                "suggest_reason": "Need it",
            }),
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown plugin"));
    }

    #[test]
    fn request_plugin_install_denied_skips_install() {
        let fixture = catalog_with_demo_plugin();
        let ctx = test_ctx(
            Arc::new(LocalToolSession::default().with_plugin_install(|_, _, _| Ok(false))),
            fixture.catalog.clone(),
        );
        let out = request_plugin_install_impl(
            &ctx,
            serde_json::json!({
                "tool_id": fixture.plugin_id,
                "suggest_reason": "Need it",
            }),
        )
        .unwrap();
        assert!(!out.is_error);
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        assert_eq!(payload["user_confirmed"], false);
        assert_eq!(payload["tools_registered"], 0);
        assert!(
            !fixture
                .catalog
                .lock()
                .unwrap()
                .is_plugin_installed(&fixture.plugin_id)
        );
    }

    #[test]
    fn request_plugin_install_confirmed_registers_tools() {
        let fixture = catalog_with_demo_plugin();
        let ctx = test_ctx(
            Arc::new(LocalToolSession::default().with_plugin_install(|_, _, _| Ok(true))),
            fixture.catalog.clone(),
        );
        let out = request_plugin_install_impl(
            &ctx,
            serde_json::json!({
                "plugin_id": fixture.plugin_id,
                "reason": "Need demo tools",
            }),
        )
        .unwrap();
        assert!(!out.is_error);
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        assert_eq!(payload["user_confirmed"], true);
        assert_eq!(payload["tools_registered"], 1);
        assert!(
            fixture
                .catalog
                .lock()
                .unwrap()
                .is_plugin_installed(&fixture.plugin_id)
        );
    }

    #[test]
    fn request_plugin_install_requires_session() {
        let fixture = catalog_with_demo_plugin();
        let ctx = test_ctx(Arc::new(LocalToolSession::default()), fixture.catalog);
        let err = request_plugin_install_impl(
            &ctx,
            serde_json::json!({
                "tool_id": fixture.plugin_id,
                "suggest_reason": "Need it",
            }),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not configured"));
    }
}

#[cfg(test)]
mod list_available_plugins_tests {
    use super::*;
    use crate::approval::{ApprovalCache, new_exec_policy};
    use crate::context::LocalToolSession;
    use crate::discovery::new_tool_catalog;
    use crate::unified_exec::UnifiedExecManager;
    use std::fs;
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    static CODEX_HOME_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct CodexHomeEnv {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl CodexHomeEnv {
        fn set(home: &Path, discover_all: bool) -> Self {
            let lock = CODEX_HOME_ENV_LOCK.lock().unwrap();
            unsafe {
                std::env::set_var("CODEX_HOME", home);
                if discover_all {
                    std::env::set_var("FORGE_PLUGIN_DISCOVER_ALL", "1");
                } else {
                    std::env::remove_var("FORGE_PLUGIN_DISCOVER_ALL");
                }
            }
            Self { _lock: lock }
        }
    }

    impl Drop for CodexHomeEnv {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("CODEX_HOME");
                std::env::remove_var("FORGE_PLUGIN_DISCOVER_ALL");
            }
        }
    }

    fn test_ctx(catalog: Arc<std::sync::Mutex<crate::discovery::ToolCatalog>>) -> ToolCallContext {
        ToolCallContext::new(
            ".",
            "off",
            Arc::new(LocalToolSession::default()),
            Arc::new(UnifiedExecManager::new()),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            catalog,
        )
    }

    fn scaffold_marketplace(codex_home: &Path) -> String {
        let agents = codex_home.join(".agents").join("plugins");
        fs::create_dir_all(&agents).unwrap();
        let plugin_src = agents.join("demo");
        let manifest_dir = plugin_src.join(".codex-plugin");
        fs::create_dir_all(&manifest_dir).unwrap();
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
                "name": "demo",
                "version": "1.0.0",
                "description": "Demo plugin",
                "mcpServers": ".mcp.json"
            }"#,
        )
        .unwrap();
        fs::write(
            plugin_src.join(".mcp.json"),
            r#"{"mcpServers": {"demo-mcp": {"command": "echo"}}}"#,
        )
        .unwrap();
        fs::write(
            agents.join("marketplace.json"),
            r#"{
                "name": "local-dev",
                "plugins": [{
                    "name": "demo",
                    "source": {"source": "local", "path": "./demo"},
                    "policy": {"installation": "AVAILABLE"}
                }]
            }"#,
        )
        .unwrap();
        "demo@local-dev".to_string()
    }

    #[test]
    fn list_available_plugins_codex_wire() {
        let dir = TempDir::new().unwrap();
        let _env = CodexHomeEnv::set(dir.path(), true);
        let plugin_id = scaffold_marketplace(dir.path());
        let ctx = test_ctx(new_tool_catalog());
        let out = list_plugins_impl(&ctx).unwrap();
        assert!(!out.is_error);
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        let tools = payload["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1, "expected one installable plugin");
        let entry = &tools[0];
        assert_eq!(entry["tool_type"], "plugin");
        assert_eq!(entry["id"], plugin_id);
        let keys: std::collections::BTreeSet<_> = entry
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "app_connector_ids",
                "description",
                "has_skills",
                "id",
                "mcp_server_names",
                "name",
                "tool_type",
            ]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
        );
    }

    #[test]
    fn list_available_plugins_empty_catalog() {
        let dir = TempDir::new().unwrap();
        let _env = CodexHomeEnv::set(dir.path(), false);
        let ctx = test_ctx(new_tool_catalog());
        let out = list_plugins_impl(&ctx).unwrap();
        assert!(!out.is_error);
        let payload: Value = serde_json::from_str(&out.content).unwrap();
        assert!(payload["tools"].as_array().unwrap().is_empty());
    }
}
