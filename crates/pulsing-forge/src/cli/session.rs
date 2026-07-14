//! REPL ToolSession — wires exec/permissions approval to stdin (Codex-style).

use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::approval::{
    ApprovalPolicy, ExecApprovalRequest, RequestPermissionsArgs, RequestPermissionsResponse,
    ReviewDecision,
};
use crate::context::{LocalToolSession, PlanItem, ToolSession, UpdatePlanArgs};
use crate::error::ToolError;
use crate::exec_output::ExecOutputDelta;
use serde_json::{Value, json};

pub struct ReplToolSession {
    inner: LocalToolSession,
    approve_auto: Arc<AtomicBool>,
}

impl ReplToolSession {
    pub fn new(approve_auto: bool) -> Arc<Self> {
        let flag = Arc::new(AtomicBool::new(approve_auto));
        let flag_exec = flag.clone();
        let flag_perm = flag.clone();
        let flag_input = flag.clone();
        let flag_plugin = flag.clone();

        let inner = LocalToolSession::default()
            .with_exec_approval(move |req| prompt_exec(&flag_exec, req))
            .with_request_permissions(move |args| prompt_permissions(&flag_perm, args))
            .with_user_input(move |args| prompt_user_input(&flag_input, args))
            .with_plugin_install(move |tool_id, tool_name, suggest_reason| {
                prompt_plugin(&flag_plugin, &tool_id, &tool_name, &suggest_reason)
            });

        Arc::new(Self {
            inner,
            approve_auto: flag,
        })
    }

    pub fn set_approve_auto(&self, auto: bool) {
        self.approve_auto.store(auto, Ordering::Relaxed);
    }

    pub fn approve_auto(&self) -> bool {
        self.approve_auto.load(Ordering::Relaxed)
    }

    pub fn plan_snapshot(&self) -> Vec<PlanItem> {
        self.inner.plan_snapshot()
    }

    pub fn new_context_requested(&self) -> bool {
        self.inner.new_context_requested()
    }

    pub fn exec_deltas(&self) -> Vec<ExecOutputDelta> {
        self.inner.exec_deltas()
    }

    pub fn tokens_remaining(&self) -> Option<i64> {
        self.inner.tokens_remaining()
    }
}

impl ToolSession for ReplToolSession {
    fn update_plan(&self, args: UpdatePlanArgs) -> Result<(), ToolError> {
        self.inner.update_plan(args)
    }

    fn request_new_context(&self) -> Result<(), ToolError> {
        self.inner.request_new_context()
    }

    fn tokens_remaining(&self) -> Option<i64> {
        self.inner.tokens_remaining()
    }

    fn request_user_input(&self, arguments: Value) -> Result<Value, ToolError> {
        self.inner.request_user_input(arguments)
    }

    fn on_exec_output_delta(&self, delta: ExecOutputDelta) -> Result<(), ToolError> {
        self.inner.on_exec_output_delta(delta)
    }

    fn approval_policy(&self) -> ApprovalPolicy {
        if self.approve_auto.load(Ordering::Relaxed) {
            ApprovalPolicy::Always
        } else {
            ApprovalPolicy::OnRequest
        }
    }

    fn request_exec_approval(
        &self,
        request: ExecApprovalRequest,
    ) -> Result<ReviewDecision, ToolError> {
        self.inner.request_exec_approval(request)
    }

    fn request_permissions(
        &self,
        args: RequestPermissionsArgs,
    ) -> Result<RequestPermissionsResponse, ToolError> {
        self.inner.request_permissions(args)
    }

    fn request_plugin_install(
        &self,
        tool_id: String,
        tool_name: String,
        suggest_reason: String,
    ) -> Result<bool, ToolError> {
        self.inner
            .request_plugin_install(tool_id, tool_name, suggest_reason)
    }
}

fn prompt_exec(
    approve_auto: &AtomicBool,
    req: ExecApprovalRequest,
) -> Result<ReviewDecision, ToolError> {
    if approve_auto.load(Ordering::Relaxed) {
        return Ok(ReviewDecision::Approved);
    }
    let cmd = req.command.join(" ");
    eprintln!("\n── exec approval ──");
    eprintln!("command: {cmd}");
    if let Some(reason) = &req.reason {
        eprintln!("reason: {reason}");
    }
    eprint!("  [y] once  [a] session  [n] deny  > ");
    let _ = io::stderr().flush();
    let choice = read_stdin_line()?;
    match choice.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Ok(ReviewDecision::Approved),
        "a" | "allow" | "session" => Ok(ReviewDecision::ApprovedForSession),
        _ => Ok(ReviewDecision::Denied),
    }
}

fn prompt_permissions(
    approve_auto: &AtomicBool,
    args: RequestPermissionsArgs,
) -> Result<RequestPermissionsResponse, ToolError> {
    if approve_auto.load(Ordering::Relaxed) {
        return Ok(RequestPermissionsResponse {
            permissions: args.permissions,
            scope: "session".into(),
            strict_auto_review: false,
        });
    }
    eprintln!("\n── permissions request ──");
    if let Some(reason) = &args.reason {
        eprintln!("reason: {reason}");
    }
    eprint!("grant permissions? [y/N/a=session] > ");
    let _ = io::stderr().flush();
    let choice = read_stdin_line()?;
    match choice.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" | "a" | "allow" => Ok(RequestPermissionsResponse {
            permissions: args.permissions,
            scope: if choice.starts_with('a') {
                "session".into()
            } else {
                "once".into()
            },
            strict_auto_review: false,
        }),
        _ => Err(ToolError::respond("permissions denied by user")),
    }
}

fn prompt_user_input(approve_auto: &AtomicBool, args: Value) -> Result<Value, ToolError> {
    if approve_auto.load(Ordering::Relaxed) {
        if let Some(questions) = args.get("questions").and_then(|q| q.as_array())
            && let Some(q0) = questions.first()
        {
            let id = q0.get("id").and_then(|v| v.as_str()).unwrap_or("q0");
            if let Some(opts) = q0.get("options").and_then(|o| o.as_array())
                && let Some(opt) = opts.first()
            {
                let label = opt.get("label").and_then(|v| v.as_str()).unwrap_or("yes");
                return Ok(json!({ "answers": { id: label } }));
            }
        }
        return Ok(json!({ "answers": {} }));
    }
    eprintln!("\n── user input request ──");
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&args).unwrap_or_default()
    );
    eprint!("accept first option? [Y/n/custom JSON] > ");
    let _ = io::stderr().flush();
    let line = read_stdin_line()?;
    let trimmed = line.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("y")
        || trimmed.eq_ignore_ascii_case("yes")
    {
        return prompt_user_input(&AtomicBool::new(true), args);
    }
    if trimmed.eq_ignore_ascii_case("n") || trimmed.eq_ignore_ascii_case("no") {
        return Ok(json!({ "answers": {} }));
    }
    serde_json::from_str(trimmed).map_err(|e| ToolError::respond(format!("invalid JSON: {e}")))
}

fn prompt_plugin(
    approve_auto: &AtomicBool,
    tool_id: &str,
    tool_name: &str,
    suggest_reason: &str,
) -> Result<bool, ToolError> {
    if approve_auto.load(Ordering::Relaxed) {
        return Ok(true);
    }
    eprint!("\ninstall plugin {tool_name} ({tool_id})? reason: {suggest_reason} [y/N] > ");
    let _ = io::stderr().flush();
    let line = read_stdin_line()?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes" | "allow"
    ))
}

fn read_stdin_line() -> Result<String, ToolError> {
    let mut buf = String::new();
    io::stdin()
        .read_line(&mut buf)
        .map_err(|e| ToolError::respond(format!("stdin: {e}")))?;
    Ok(buf)
}
