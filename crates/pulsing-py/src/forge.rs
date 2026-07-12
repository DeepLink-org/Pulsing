//! PyO3 bindings for `pulsing-forge` — Rust-native tool runtime.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use pulsing_forge::approval::{
    ApprovalPolicy, ExecApprovalRequest, RequestPermissionsArgs, RequestPermissionsResponse,
    ReviewDecision,
};
use pulsing_forge::context::{LocalToolSession, NullToolSession, ToolSession, UpdatePlanArgs};
use pulsing_forge::error::ToolError;
use pulsing_forge::exec_output::ExecOutputDelta;
use pulsing_forge::mcp::{new_shared_mcp_runtime, refresh_mcp_runtime, SharedMcpRuntime};
use pulsing_forge::result::ToolResult;
use pulsing_forge::runtime::{ToolRuntime, ToolRuntimeConfig};
use pulsing_forge::unified_exec::UnifiedExecManager;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::Value;
use tokio::runtime::Handle;

struct PyForgeSession {
    event_cb: Py<PyAny>,
    user_input_cb: Option<Py<PyAny>>,
    exec_approval_cb: Option<Py<PyAny>>,
    permissions_cb: Option<Py<PyAny>>,
    tokens_remaining_cb: Option<Py<PyAny>>,
    plugin_install_cb: Option<Py<PyAny>>,
    auto_approve: bool,
}

unsafe impl Send for PyForgeSession {}
unsafe impl Sync for PyForgeSession {}

impl PyForgeSession {
    fn new(
        event_cb: Py<PyAny>,
        user_input_cb: Option<Py<PyAny>>,
        exec_approval_cb: Option<Py<PyAny>>,
        permissions_cb: Option<Py<PyAny>>,
        tokens_remaining_cb: Option<Py<PyAny>>,
        plugin_install_cb: Option<Py<PyAny>>,
        auto_approve: bool,
    ) -> Self {
        Self {
            event_cb,
            user_input_cb,
            exec_approval_cb,
            permissions_cb,
            tokens_remaining_cb,
            plugin_install_cb,
            auto_approve,
        }
    }

    fn emit(
        &self,
        py: Python<'_>,
        kind: &str,
        payload: Value,
        source: Option<&str>,
    ) -> Result<(), ToolError> {
        let dict = PyDict::new(py);
        dict.set_item("kind", kind)
            .map_err(|e| ToolError::respond(e.to_string()))?;
        dict.set_item(
            "payload",
            json_to_py(py, &payload).map_err(|e| ToolError::respond(e.to_string()))?,
        )
        .map_err(|e| ToolError::respond(e.to_string()))?;
        if let Some(s) = source {
            dict.set_item("source", s)
                .map_err(|e| ToolError::respond(e.to_string()))?;
        }
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        dict.set_item("ts", ts)
            .map_err(|e| ToolError::respond(e.to_string()))?;
        self.event_cb
            .call1(py, (dict,))
            .map_err(|e| ToolError::respond(e.to_string()))?;
        Ok(())
    }
}

impl ToolSession for PyForgeSession {
    fn update_plan(&self, args: UpdatePlanArgs) -> Result<(), ToolError> {
        Python::with_gil(|py| {
            let plan: Vec<Value> = args
                .plan
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "step": p.step,
                        "status": match p.status {
                            pulsing_forge::context::StepStatus::Pending => "pending",
                            pulsing_forge::context::StepStatus::InProgress => "in_progress",
                            pulsing_forge::context::StepStatus::Completed => "completed",
                        }
                    })
                })
                .collect();
            self.emit(
                py,
                "plan_updated",
                serde_json::json!({ "plan": plan }),
                None,
            )
        })
    }

    fn request_new_context(&self) -> Result<(), ToolError> {
        Python::with_gil(|py| self.emit(py, "new_context", serde_json::json!({}), None))
    }

    fn tokens_remaining(&self) -> Option<i64> {
        Python::with_gil(|py| {
            let cb = self.tokens_remaining_cb.as_ref()?;
            let out = match cb.call0(py) {
                Ok(out) => out,
                Err(e) => {
                    tracing::warn!("tokens_remaining callback raised: {e}");
                    return None;
                }
            };
            if out.is_none(py) {
                return None;
            }
            match out.extract::<i64>(py) {
                Ok(n) => Some(n),
                Err(e) => {
                    tracing::warn!("tokens_remaining callback returned non-int: {e}");
                    None
                }
            }
        })
    }

    fn request_user_input(&self, arguments: Value) -> Result<Value, ToolError> {
        Python::with_gil(|py| {
            self.emit(py, "user_input_request", arguments.clone(), None)?;
            let cb = self.user_input_cb.as_ref().ok_or_else(|| {
                ToolError::respond("request_user_input is not configured on this ToolSession")
            })?;
            let py_args =
                json_to_py(py, &arguments).map_err(|e| ToolError::respond(e.to_string()))?;
            let out = cb
                .call1(py, (py_args,))
                .map_err(|e| ToolError::respond(e.to_string()))?;
            py_to_json(&out.bind(py)).map_err(|e| ToolError::respond(e.to_string()))
        })
    }

    fn on_exec_output_delta(&self, delta: ExecOutputDelta) -> Result<(), ToolError> {
        Python::with_gil(|py| {
            let stream = match delta.stream {
                pulsing_forge::exec_output::ExecStream::Stdout => "stdout",
                pulsing_forge::exec_output::ExecStream::Stderr => "stderr",
                pulsing_forge::exec_output::ExecStream::Pty => "pty",
            };
            self.emit(
                py,
                "exec_output_delta",
                serde_json::json!({
                    "session_id": delta.session_id,
                    "stream": stream,
                    "chunk": delta.chunk,
                }),
                None,
            )
        })
    }

    fn approval_policy(&self) -> ApprovalPolicy {
        if self.auto_approve {
            ApprovalPolicy::Always
        } else {
            ApprovalPolicy::OnRequest
        }
    }

    fn request_exec_approval(
        &self,
        request: ExecApprovalRequest,
    ) -> Result<ReviewDecision, ToolError> {
        if self.auto_approve {
            return Ok(ReviewDecision::Approved);
        }
        Python::with_gil(|py| {
            let payload =
                serde_json::to_value(&request).map_err(|e| ToolError::respond(e.to_string()))?;
            self.emit(py, "exec_approval_request", payload.clone(), None)?;
            let cb = self.exec_approval_cb.as_ref().ok_or_else(|| {
                ToolError::respond("exec approval is not configured on this ToolSession")
            })?;
            let py_req = json_to_py(py, &payload).map_err(|e| ToolError::respond(e.to_string()))?;
            let out = cb
                .call1(py, (py_req,))
                .map_err(|e| ToolError::respond(e.to_string()))?;
            parse_review_decision(&out.bind(py), &request)
        })
    }

    fn request_permissions(
        &self,
        args: RequestPermissionsArgs,
    ) -> Result<RequestPermissionsResponse, ToolError> {
        if self.auto_approve {
            return Ok(RequestPermissionsResponse {
                permissions: args.permissions.clone(),
                scope: "session".into(),
                strict_auto_review: false,
            });
        }
        Python::with_gil(|py| {
            let payload =
                serde_json::to_value(&args).map_err(|e| ToolError::respond(e.to_string()))?;
            self.emit(py, "request_permissions", payload.clone(), None)?;
            let cb = self.permissions_cb.as_ref().ok_or_else(|| {
                ToolError::respond("request_permissions is not configured on this ToolSession")
            })?;
            let py_args =
                json_to_py(py, &payload).map_err(|e| ToolError::respond(e.to_string()))?;
            let out = cb
                .call1(py, (py_args,))
                .map_err(|e| ToolError::respond(e.to_string()))?;
            serde_json::from_value(
                py_to_json(&out.bind(py)).map_err(|e| ToolError::respond(e.to_string()))?,
            )
            .map_err(|e| ToolError::respond(format!("invalid request_permissions response: {e}")))
        })
    }

    fn request_plugin_install(
        &self,
        tool_id: String,
        tool_name: String,
        suggest_reason: String,
    ) -> Result<bool, ToolError> {
        if self.auto_approve {
            return Ok(true);
        }
        Python::with_gil(|py| {
            let payload = serde_json::json!({
                "tool_id": tool_id,
                "tool_name": tool_name,
                "suggest_reason": suggest_reason,
            });
            self.emit(py, "plugin_install_request", payload.clone(), None)?;
            let cb = self.plugin_install_cb.as_ref().ok_or_else(|| {
                ToolError::respond("request_plugin_install is not configured on this ToolSession")
            })?;
            let py_args =
                json_to_py(py, &payload).map_err(|e| ToolError::respond(e.to_string()))?;
            let out = cb
                .call1(py, (py_args,))
                .map_err(|e| ToolError::respond(e.to_string()))?;
            if let Ok(b) = out.extract::<bool>(py) {
                return Ok(b);
            }
            if let Ok(s) = out.extract::<String>(py) {
                return Ok(matches!(
                    s.trim().to_lowercase().as_str(),
                    "approved" | "allow" | "once" | "yes" | "true"
                ));
            }
            Err(ToolError::respond(
                "plugin_install callback returned invalid decision",
            ))
        })
    }
}

fn noop_event_callback(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(py.eval_bound("lambda _event: None", None, None)?.unbind())
}

fn needs_py_forge_session(
    event_callback: &Option<Py<PyAny>>,
    user_input_callback: &Option<Py<PyAny>>,
    exec_approval_callback: &Option<Py<PyAny>>,
    request_permissions_callback: &Option<Py<PyAny>>,
    tokens_remaining_callback: &Option<Py<PyAny>>,
    plugin_install_callback: &Option<Py<PyAny>>,
) -> bool {
    event_callback.is_some()
        || user_input_callback.is_some()
        || exec_approval_callback.is_some()
        || request_permissions_callback.is_some()
        || tokens_remaining_callback.is_some()
        || plugin_install_callback.is_some()
}

#[pyclass(name = "ForgeRuntime")]
pub struct PyForgeRuntime {
    runtime: ToolRuntime,
    event_cb: Option<Py<PyAny>>,
    mcp_slot: Option<SharedMcpRuntime>,
}

#[pymethods]
impl PyForgeRuntime {
    #[new]
    #[pyo3(signature = (cwd=".", sandbox_policy="off", dangerously_disable_sandbox=false, auto_approve=false, event_callback=None, user_input_callback=None, exec_approval_callback=None, request_permissions_callback=None, tokens_remaining_callback=None, plugin_install_callback=None, start_mcp=true))]
    fn new(
        cwd: &str,
        sandbox_policy: &str,
        dangerously_disable_sandbox: bool,
        auto_approve: bool,
        event_callback: Option<Py<PyAny>>,
        user_input_callback: Option<Py<PyAny>>,
        exec_approval_callback: Option<Py<PyAny>>,
        request_permissions_callback: Option<Py<PyAny>>,
        tokens_remaining_callback: Option<Py<PyAny>>,
        plugin_install_callback: Option<Py<PyAny>>,
        start_mcp: bool,
    ) -> PyResult<Self> {
        let session: Arc<dyn ToolSession> = if needs_py_forge_session(
            &event_callback,
            &user_input_callback,
            &exec_approval_callback,
            &request_permissions_callback,
            &tokens_remaining_callback,
            &plugin_install_callback,
        ) {
            let event_cb = match &event_callback {
                Some(cb) => cb.clone(),
                None => Python::with_gil(noop_event_callback)?,
            };
            Arc::new(PyForgeSession::new(
                event_cb,
                user_input_callback,
                exec_approval_callback,
                request_permissions_callback,
                tokens_remaining_callback,
                plugin_install_callback,
                auto_approve,
            ))
        } else if auto_approve {
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always))
        } else {
            Arc::new(NullToolSession)
        };
        let mcp_slot = if start_mcp {
            Some(new_shared_mcp_runtime())
        } else {
            None
        };
        if let Some(ref slot) = mcp_slot {
            block_on_tool(refresh_mcp_runtime(slot));
        }
        let runtime = ToolRuntime::new(ToolRuntimeConfig {
            cwd: cwd.into(),
            sandbox_policy: sandbox_policy.to_string(),
            dangerously_disable_sandbox,
            session,
            exec: Arc::new(UnifiedExecManager::new()),
            mcp_runtime: mcp_slot.clone(),
            ..Default::default()
        });
        Ok(Self {
            runtime,
            event_cb: event_callback,
            mcp_slot,
        })
    }

    /// Reload MCP catalog and reconnect configured servers.
    fn refresh_mcp(&self) -> PyResult<()> {
        if let Some(slot) = &self.mcp_slot {
            block_on_tool(refresh_mcp_runtime(slot));
        }
        Ok(())
    }

    fn mcp_tool_names(&self) -> PyResult<Vec<String>> {
        if let Some(slot) = &self.mcp_slot {
            let names = block_on_tool(async {
                let guard = slot.read().await;
                guard
                    .as_ref()
                    .map(|runtime| runtime.tool_model_names())
                    .unwrap_or_default()
            });
            Ok(names)
        } else {
            Ok(vec![])
        }
    }

    /// Model-visible MCP function tools (name, description, input_schema, server/tool ids).
    fn mcp_tool_specs(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            if let Some(slot) = &self.mcp_slot {
                let specs = block_on_tool(async {
                    let guard = slot.read().await;
                    guard
                        .as_ref()
                        .map(|runtime| runtime.tool_specs_for_model())
                        .unwrap_or_default()
                });
                specs
                    .iter()
                    .map(|spec| json_to_py(py, spec).map(|v| v.into()))
                    .collect()
            } else {
                Ok(vec![])
            }
        })
    }

    fn tool_names(&self) -> Vec<String> {
        self.runtime.tool_names()
    }

    /// Execute a tool (sync — for actor workers and in-process hosts).
    #[pyo3(signature = (name, arguments=None))]
    fn call_tool(
        &self,
        py: Python<'_>,
        name: String,
        arguments: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let args = match arguments {
            Some(obj) => py_to_json(&obj)?,
            None => Value::Object(Default::default()),
        };
        if let Some(ref cb) = self.event_cb {
            let dict = PyDict::new(py);
            dict.set_item("kind", "tool_begin")?;
            let payload = PyDict::new(py);
            payload.set_item("arguments", json_to_py(py, &args)?)?;
            dict.set_item("payload", payload)?;
            dict.set_item("source", &name)?;
            cb.call1(py, (dict,))?;
        }
        let name_for = name.clone();
        let out = py.allow_threads(|| {
            block_on_tool(async { self.runtime.call_tool(&name_for, args).await })
        });
        if let Some(ref cb) = self.event_cb {
            let dict = PyDict::new(py);
            dict.set_item("kind", "tool_end")?;
            let payload = PyDict::new(py);
            payload.set_item("is_error", out.is_error)?;
            let preview: String = out.content.chars().take(500).collect();
            payload.set_item("content_preview", preview)?;
            dict.set_item("payload", payload)?;
            dict.set_item("source", &name)?;
            cb.call1(py, (dict,))?;
        }
        tool_result_to_py(py, &out)
    }
}

fn parse_review_decision(
    obj: &Bound<'_, PyAny>,
    request: &ExecApprovalRequest,
) -> Result<ReviewDecision, ToolError> {
    if let Ok(s) = obj.extract::<String>() {
        return map_decision_str(&s, request);
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        if let Some(item) = dict.get_item("decision").ok().flatten() {
            if let Ok(s) = item.extract::<String>() {
                return map_decision_str(&s, request);
            }
        }
    }
    Err(ToolError::respond(
        "exec approval callback returned invalid decision",
    ))
}

fn map_decision_str(raw: &str, request: &ExecApprovalRequest) -> Result<ReviewDecision, ToolError> {
    match raw.trim().to_lowercase().as_str() {
        "approved" | "allow" | "once" => Ok(ReviewDecision::Approved),
        "approved_for_session" | "allow_session" => Ok(ReviewDecision::ApprovedForSession),
        "approved_with_amendment" | "allow_always" => {
            let prefix = request
                .proposed_execpolicy_amendment
                .clone()
                .unwrap_or_else(|| request.command.clone());
            Ok(ReviewDecision::ApprovedExecpolicyAmendment {
                proposed_execpolicy_amendment: prefix,
            })
        }
        "abort" => Ok(ReviewDecision::Abort),
        _ => Ok(ReviewDecision::Denied),
    }
}

fn block_on_tool<F, T>(f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    if let Ok(handle) = Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(f))
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("forge tokio runtime")
            .block_on(f)
    }
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = obj.py();
    let json = py.import("json")?;
    let s: String = json.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&s).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    let json = py.import("json")?;
    let s = serde_json::to_string(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(json.call_method1("loads", (s,))?)
}

fn tool_result_to_py(py: Python<'_>, r: &ToolResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("content", &r.content)?;
    dict.set_item("is_error", r.is_error)?;
    if let Some(ref structured) = r.structured {
        dict.set_item("structured", json_to_py(py, structured)?)?;
    }
    Ok(dict.into())
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForgeRuntime>()?;
    Ok(())
}
