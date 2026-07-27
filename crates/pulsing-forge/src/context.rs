//! Host-facing abstractions for session/plan tools (framework-agnostic).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::approval::{
    ApprovalCache, ApprovalPolicy, ExecApprovalRequest, RequestPermissionsArgs,
    RequestPermissionsResponse, ReviewDecision,
};
use crate::discovery::ToolCatalog;
use crate::execpolicy::ExecPolicy;

use crate::error::ToolError;
use crate::exec_output::ExecOutputDelta;
use crate::sandbox::{SandboxPolicy, normalize_policy};
use crate::unified_exec::UnifiedExecManager;

/// Plan step status for collaborative plan tools.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
}

/// Single plan item for `update_plan`.
/// Mirrors Codex's `PlanItemArg` (vendor/codex-rs/protocol/src/plan_tool.rs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanItem {
    pub step: String,
    pub status: StepStatus,
}

/// Arguments for the `update_plan` tool.
/// Mirrors Codex's `UpdatePlanArgs` (vendor/codex-rs/protocol/src/plan_tool.rs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdatePlanArgs {
    #[serde(default)]
    pub explanation: Option<String>,
    pub plan: Vec<PlanItem>,
}

/// Callback host for plan/session tools. Implementations live in Craft, CLI, tests, etc.
pub trait ToolSession: Send + Sync {
    fn update_plan(&self, args: UpdatePlanArgs) -> Result<(), ToolError>;

    fn request_new_context(&self) -> Result<(), ToolError>;

    fn tokens_remaining(&self) -> Option<i64>;

    /// Host shows UI and returns structured answers JSON.
    fn request_user_input(
        &self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError>;

    /// Streaming unified-exec output (Codex `ExecCommandOutputDelta` equivalent).
    fn on_exec_output_delta(&self, _delta: ExecOutputDelta) -> Result<(), ToolError> {
        Ok(())
    }

    /// Host approval policy for shell / exec (Codex `AskForApproval` subset).
    fn approval_policy(&self) -> ApprovalPolicy {
        ApprovalPolicy::OnRequest
    }

    /// Prompt user before running a shell command (Codex ExecApprovalRequest).
    fn request_exec_approval(
        &self,
        _request: ExecApprovalRequest,
    ) -> Result<ReviewDecision, ToolError> {
        Err(ToolError::respond(
            "exec approval is not configured on this ToolSession",
        ))
    }

    /// Resolve `request_permissions` tool (Codex permission escalation).
    fn request_permissions(
        &self,
        _args: RequestPermissionsArgs,
    ) -> Result<RequestPermissionsResponse, ToolError> {
        Err(ToolError::respond(
            "request_permissions is not configured on this ToolSession",
        ))
    }

    /// Prompt before installing a Codex-compatible plugin.
    fn request_plugin_install(
        &self,
        _tool_id: String,
        _tool_name: String,
        _suggest_reason: String,
    ) -> Result<bool, ToolError> {
        Err(ToolError::respond(
            "request_plugin_install is not configured on this ToolSession",
        ))
    }
}

/// Per-invocation context passed to every handler.
#[derive(Clone)]
pub struct ToolCallContext {
    pub cwd: PathBuf,
    pub sandbox_policy: SandboxPolicy,
    pub dangerously_disable_sandbox: bool,
    pub session: Arc<dyn ToolSession>,
    pub exec: Arc<UnifiedExecManager>,
    pub exec_policy: Arc<Mutex<ExecPolicy>>,
    pub approval_cache: Arc<ApprovalCache>,
    pub tool_catalog: Arc<Mutex<ToolCatalog>>,
    pub mcp_runtime: Option<crate::mcp::SharedMcpRuntime>,
    pub turn: Option<Arc<crate::turn::TurnExecutionContext>>,
}

impl ToolCallContext {
    pub fn new(
        cwd: impl AsRef<Path>,
        sandbox_policy: &str,
        session: Arc<dyn ToolSession>,
        exec: Arc<UnifiedExecManager>,
        exec_policy: Arc<Mutex<ExecPolicy>>,
        approval_cache: Arc<ApprovalCache>,
        tool_catalog: Arc<Mutex<ToolCatalog>>,
    ) -> Self {
        Self {
            cwd: cwd.as_ref().to_path_buf(),
            sandbox_policy: normalize_policy(sandbox_policy),
            dangerously_disable_sandbox: false,
            session,
            exec,
            exec_policy,
            approval_cache,
            tool_catalog,
            mcp_runtime: None,
            turn: None,
        }
    }

    pub fn with_mcp_runtime(mut self, mcp: crate::mcp::SharedMcpRuntime) -> Self {
        self.mcp_runtime = Some(mcp);
        self
    }

    pub fn with_dangerous_sandbox(mut self, disable: bool) -> Self {
        self.dangerously_disable_sandbox = disable;
        self
    }

    pub fn with_turn(mut self, turn: Arc<crate::turn::TurnExecutionContext>) -> Self {
        self.turn = Some(turn);
        self
    }
}

/// In-memory session for local runs and tests.
pub struct LocalToolSession {
    plan: std::sync::Mutex<Vec<PlanItem>>,
    new_context_requested: std::sync::Mutex<bool>,
    tokens_remaining: Option<i64>,
    user_input: Option<
        Arc<dyn Fn(serde_json::Value) -> Result<serde_json::Value, ToolError> + Send + Sync>,
    >,
    exec_deltas: std::sync::Mutex<Vec<ExecOutputDelta>>,
    exec_approval:
        Option<Arc<dyn Fn(ExecApprovalRequest) -> Result<ReviewDecision, ToolError> + Send + Sync>>,
    permissions: Option<
        Arc<
            dyn Fn(RequestPermissionsArgs) -> Result<RequestPermissionsResponse, ToolError>
                + Send
                + Sync,
        >,
    >,
    plugin_install:
        Option<Arc<dyn Fn(String, String, String) -> Result<bool, ToolError> + Send + Sync>>,
    approval_policy: ApprovalPolicy,
}

impl Default for LocalToolSession {
    fn default() -> Self {
        Self {
            plan: std::sync::Mutex::new(Vec::new()),
            new_context_requested: std::sync::Mutex::new(false),
            tokens_remaining: None,
            user_input: None,
            exec_deltas: std::sync::Mutex::new(Vec::new()),
            exec_approval: None,
            permissions: None,
            plugin_install: None,
            approval_policy: ApprovalPolicy::OnRequest,
        }
    }
}

impl LocalToolSession {
    pub fn new(tokens_remaining: Option<i64>) -> Self {
        Self {
            tokens_remaining,
            ..Default::default()
        }
    }

    pub fn with_user_input<F>(mut self, f: F) -> Self
    where
        F: Fn(serde_json::Value) -> Result<serde_json::Value, ToolError> + Send + Sync + 'static,
    {
        self.user_input = Some(Arc::new(f));
        self
    }

    pub fn with_exec_approval<F>(mut self, f: F) -> Self
    where
        F: Fn(ExecApprovalRequest) -> Result<ReviewDecision, ToolError> + Send + Sync + 'static,
    {
        self.exec_approval = Some(Arc::new(f));
        self
    }

    pub fn with_request_permissions<F>(mut self, f: F) -> Self
    where
        F: Fn(RequestPermissionsArgs) -> Result<RequestPermissionsResponse, ToolError>
            + Send
            + Sync
            + 'static,
    {
        self.permissions = Some(Arc::new(f));
        self
    }

    pub fn with_plugin_install<F>(mut self, f: F) -> Self
    where
        F: Fn(String, String, String) -> Result<bool, ToolError> + Send + Sync + 'static,
    {
        self.plugin_install = Some(Arc::new(f));
        self
    }

    pub fn with_approval_policy(mut self, policy: ApprovalPolicy) -> Self {
        self.approval_policy = policy;
        self
    }

    pub fn plan_snapshot(&self) -> Vec<PlanItem> {
        self.plan.lock().unwrap().clone()
    }

    pub fn new_context_requested(&self) -> bool {
        *self.new_context_requested.lock().unwrap()
    }

    pub fn exec_deltas(&self) -> Vec<ExecOutputDelta> {
        self.exec_deltas.lock().unwrap().clone()
    }
}

impl ToolSession for LocalToolSession {
    fn update_plan(&self, args: UpdatePlanArgs) -> Result<(), ToolError> {
        *self.plan.lock().unwrap() = args.plan;
        Ok(())
    }

    fn request_new_context(&self) -> Result<(), ToolError> {
        *self.new_context_requested.lock().unwrap() = true;
        Ok(())
    }

    fn tokens_remaining(&self) -> Option<i64> {
        self.tokens_remaining
    }

    fn request_user_input(
        &self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        match &self.user_input {
            Some(f) => f(arguments),
            None => Err(ToolError::respond(
                "request_user_input is not configured on this ToolSession",
            )),
        }
    }

    fn on_exec_output_delta(&self, delta: ExecOutputDelta) -> Result<(), ToolError> {
        self.exec_deltas.lock().unwrap().push(delta);
        Ok(())
    }

    fn approval_policy(&self) -> ApprovalPolicy {
        self.approval_policy
    }

    fn request_exec_approval(
        &self,
        request: ExecApprovalRequest,
    ) -> Result<ReviewDecision, ToolError> {
        match &self.exec_approval {
            Some(f) => f(request),
            None => Err(ToolError::respond(
                "exec approval is not configured on this ToolSession",
            )),
        }
    }

    fn request_permissions(
        &self,
        args: RequestPermissionsArgs,
    ) -> Result<RequestPermissionsResponse, ToolError> {
        match &self.permissions {
            Some(f) => f(args),
            None => Err(ToolError::respond(
                "request_permissions is not configured on this ToolSession",
            )),
        }
    }

    fn request_plugin_install(
        &self,
        tool_id: String,
        tool_name: String,
        suggest_reason: String,
    ) -> Result<bool, ToolError> {
        match &self.plugin_install {
            Some(f) => f(tool_id, tool_name, suggest_reason),
            None => Err(ToolError::respond(
                "request_plugin_install is not configured on this ToolSession",
            )),
        }
    }
}

/// No-op session when host does not care about plan/session side effects.
pub struct NullToolSession;

impl ToolSession for NullToolSession {
    fn update_plan(&self, _args: UpdatePlanArgs) -> Result<(), ToolError> {
        Ok(())
    }

    fn request_new_context(&self) -> Result<(), ToolError> {
        Ok(())
    }

    fn tokens_remaining(&self) -> Option<i64> {
        None
    }

    fn request_user_input(
        &self,
        _arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        Err(ToolError::respond(
            "request_user_input is not available in this runtime",
        ))
    }
}
