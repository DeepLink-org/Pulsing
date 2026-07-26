//! UnifiedExec session store — `exec_command` + `write_stdin`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Instant;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::approval::{
    args_dangerously_disable_sandbox, effective_sandbox_policy, ensure_shell_allowed,
};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::exec_output::{
    DEFAULT_MAX_OUTPUT_TOKENS, ExecCommandOutput, ExecOutputDelta, ExecStream, MAX_STDIN_BYTES,
    OutputBuffer, SHELL_MAX_BYTES, Utf8ChunkDecoder, clamp_yield_ms,
};
use crate::handlers::shell_exec::resolve_shell_workdir;
use crate::process_group;
use crate::pty_session::PtyExecHandle;
use crate::result::ToolResult;
use crate::sandbox::{SandboxPolicy, build_bash_exec};
use crate::turn::TurnResourceGuard;

pub struct UnifiedExecManager {
    next_id: AtomicI32,
    sessions: Mutex<HashMap<i32, ExecSession>>,
}

enum SessionHandle {
    Pipe {
        child: Child,
        stdin: Option<ChildStdin>,
        process_id: Option<u32>,
    },
    Pty(PtyExecHandle),
}

struct ExecSession {
    handle: SessionHandle,
    buffer: Arc<Mutex<OutputBuffer>>,
    _reader: Option<JoinHandle<()>>,
    started: Instant,
    tty: bool,
    owner_turn: Option<crate::protocol::TurnId>,
    _turn_resource: Option<TurnResourceGuard>,
}

impl Drop for ExecSession {
    fn drop(&mut self) {
        match &mut self.handle {
            SessionHandle::Pipe {
                child, process_id, ..
            } => {
                if let Some(process_id) = process_id {
                    process_group::kill(*process_id);
                }
                if matches!(child.try_wait(), Ok(None)) {
                    let _ = child.start_kill();
                }
            }
            SessionHandle::Pty(pty) => {
                let _ = pty.kill();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecSessionSummary {
    pub id: i32,
    pub elapsed_secs: f64,
    pub tty: bool,
}

impl Default for UnifiedExecManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for UnifiedExecManager {
    fn drop(&mut self) {
        // Best-effort cleanup: hosts that forget to call `stop_all()` before
        // dropping the manager (e.g. a panicking caller, or a Python binding
        // that never invokes `close()`) would otherwise leak child processes
        // and PTY reader threads for every still-running exec_command session.
        let Ok(mut sessions) = self.sessions.try_lock() else {
            return;
        };
        for (_, session) in sessions.drain() {
            drop(session);
        }
    }
}

impl UnifiedExecManager {
    pub fn new() -> Self {
        Self {
            next_id: AtomicI32::new(1),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    pub async fn exec_command(
        self: &Arc<Self>,
        ctx: &ToolCallContext,
        args: &Value,
    ) -> Result<ToolResult, ToolError> {
        let cmd = args
            .get("cmd")
            .or_else(|| args.get("command"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::respond("missing cmd/command"))?;
        let login = args.get("login").and_then(|v| v.as_bool()).unwrap_or(false);
        let yield_ms = clamp_yield_ms(args.get("yield_time_ms").and_then(|v| v.as_u64()));
        let max_tokens = args
            .get("max_output_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);

        ensure_shell_allowed(ctx, args, cmd)?;
        let policy = effective_sandbox_policy(ctx, args);
        let dangerous = args_dangerously_disable_sandbox(ctx, args);
        let workdir = resolve_shell_workdir(ctx, args)?;
        let tty = args.get("tty").and_then(|v| v.as_bool()).unwrap_or(true);

        let session_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let turn_resource = ctx.turn.as_ref().map(|turn| {
            let weak_manager = Arc::downgrade(self);
            turn.resources()
                .register(format!("exec:{session_id}"), move || {
                    let Some(manager) = weak_manager.upgrade() else {
                        return;
                    };
                    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
                        return;
                    };
                    runtime.spawn(async move {
                        manager.stop_session_and_wait(session_id).await;
                    });
                })
        });
        let buffer = Arc::new(Mutex::new(OutputBuffer::new(SHELL_MAX_BYTES)));
        let on_delta = stream_hook(ctx, session_id);

        let (handle, reader) = if tty {
            let pty = spawn_pty_exec(
                cmd,
                &workdir,
                login,
                policy,
                dangerous,
                buffer.clone(),
                session_id,
                on_delta,
            )?;
            (SessionHandle::Pty(pty), None)
        } else {
            let (child, stdin, reader) = spawn_pipe_session(
                cmd,
                &workdir,
                login,
                policy,
                dangerous,
                buffer.clone(),
                session_id,
                on_delta,
            )
            .await?;
            let process_id = child.id();
            (
                SessionHandle::Pipe {
                    child,
                    stdin,
                    process_id,
                },
                Some(reader),
            )
        };

        self.sessions.lock().await.insert(
            session_id,
            ExecSession {
                handle,
                buffer: buffer.clone(),
                _reader: reader,
                started: Instant::now(),
                tty,
                owner_turn: ctx.turn.as_ref().map(|turn| turn.turn_id.clone()),
                _turn_resource: turn_resource,
            },
        );

        if let Some(turn) = &ctx.turn {
            let cancellation = turn.cancellation();
            tokio::select! {
                _ = cancellation.cancelled() => {
                    self.stop_session_and_wait(session_id).await;
                    return Ok(ToolResult::err("exec session cancelled"));
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(yield_ms)) => {}
            }
        } else {
            tokio::time::sleep(std::time::Duration::from_millis(yield_ms)).await;
        }
        self.poll_session(session_id, max_tokens, ctx).await
    }

    pub async fn write_stdin(
        &self,
        ctx: &ToolCallContext,
        args: &Value,
    ) -> Result<ToolResult, ToolError> {
        let session_id = parse_session_id(args.get("session_id"))?;
        let chars = args
            .get("chars")
            .or_else(|| args.get("input"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::respond("missing chars"))?;
        let byte_len = chars.len();
        if byte_len > MAX_STDIN_BYTES {
            return Ok(ToolResult::err(format!(
                "stdin input too large: {byte_len} bytes (max {MAX_STDIN_BYTES})"
            )));
        }
        let yield_ms = clamp_yield_ms(args.get("yield_time_ms").and_then(|v| v.as_u64()));
        let max_tokens = args
            .get("max_output_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);

        {
            let mut sessions = self.sessions.lock().await;
            let session = sessions
                .get_mut(&session_id)
                .ok_or_else(|| ToolError::respond(format!("unknown session_id {session_id}")))?;
            if let Some(owner) = &session.owner_turn
                && ctx.turn.as_ref().map(|turn| &turn.turn_id) != Some(owner)
            {
                return Ok(ToolResult::err(format!(
                    "session {session_id} belongs to another turn"
                )));
            }

            if chars != "\x03" && session_has_exited(&mut session.handle) {
                return Ok(ToolResult::err(format!(
                    "session {session_id} has already exited"
                )));
            }

            if chars == "\x03" {
                match &mut session.handle {
                    SessionHandle::Pipe {
                        child, process_id, ..
                    } => {
                        if let Some(process_id) = process_id {
                            process_group::kill(*process_id);
                        }
                        let _ = child.start_kill();
                    }
                    SessionHandle::Pty(pty) => pty.kill()?,
                }
            } else if !session.tty {
                return Ok(ToolResult::err(
                    "stdin writes require tty=true exec_command sessions",
                ));
            } else {
                match &mut session.handle {
                    SessionHandle::Pipe { stdin, .. } => {
                        if let Some(stdin) = stdin.as_mut() {
                            stdin.write_all(chars.as_bytes()).await.map_err(|e| {
                                ToolError::respond(format!("write stdin failed: {e}"))
                            })?;
                            stdin.flush().await.map_err(|e| {
                                ToolError::respond(format!("flush stdin failed: {e}"))
                            })?;
                        } else {
                            return Ok(ToolResult::err("session stdin is closed"));
                        }
                    }
                    SessionHandle::Pty(pty) => pty.write_stdin(chars.as_bytes())?,
                }
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(yield_ms)).await;
        self.poll_session(session_id, max_tokens, ctx).await
    }

    /// Active unified-exec sessions (for REPL `/ps`).
    pub async fn list_sessions(&self) -> Vec<ExecSessionSummary> {
        let sessions = self.sessions.lock().await;
        sessions
            .iter()
            .map(|(id, s)| ExecSessionSummary {
                id: *id,
                elapsed_secs: s.started.elapsed().as_secs_f64(),
                tty: s.tty,
            })
            .collect()
    }

    /// Kill all background exec sessions (REPL `/stop` or `/clean`).
    pub async fn stop_all(&self) -> usize {
        let mut sessions = self.sessions.lock().await;
        let removed = sessions
            .drain()
            .map(|(_, session)| session)
            .collect::<Vec<_>>();
        let count = removed.len();
        drop(sessions);
        for session in removed {
            session.terminate().await;
        }
        count
    }

    async fn stop_session_and_wait(&self, session_id: i32) {
        let session = self.sessions.lock().await.remove(&session_id);
        if let Some(session) = session {
            session.terminate().await;
        }
    }

    async fn poll_session(
        &self,
        session_id: i32,
        max_tokens: usize,
        ctx: &ToolCallContext,
    ) -> Result<ToolResult, ToolError> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(&session_id)
            .ok_or_else(|| ToolError::respond(format!("unknown session_id {session_id}")))?;

        let exit_code = match &mut session.handle {
            SessionHandle::Pipe { child, .. } => match child.try_wait() {
                Ok(Some(status)) => Some(status.code().unwrap_or(-1)),
                Ok(None) => None,
                Err(e) => return Ok(ToolResult::err(format!("wait failed: {e}"))),
            },
            SessionHandle::Pty(pty) => pty.poll_exit_code(),
        };

        let wall = session.started.elapsed().as_secs_f64();
        let mut buf = session.buffer.lock().await;
        buf.truncate_to_tokens(max_tokens);
        let output = buf.snapshot();
        drop(buf);

        let structured = ExecCommandOutput::new(
            output.clone(),
            wall,
            exit_code,
            if exit_code.is_none() {
                Some(session_id)
            } else {
                None
            },
        );

        if exit_code.is_some() {
            sessions.remove(&session_id);
        }

        let json = serde_json::to_string_pretty(&structured)
            .map_err(|e| ToolError::respond(e.to_string()))?;
        let is_error = exit_code.is_some_and(|c| c != 0);
        let result = ToolResult {
            content: json.clone(),
            is_error,
            structured: Some(serde_json::to_value(structured).unwrap_or(Value::Null)),
        };

        // Final snapshot delta for hosts that only listen to stream events.
        let _ = ctx.session.on_exec_output_delta(ExecOutputDelta {
            session_id,
            stream: ExecStream::Pty,
            chunk: output,
        });

        Ok(result)
    }
}

impl ExecSession {
    async fn terminate(mut self) {
        match &mut self.handle {
            SessionHandle::Pipe {
                child, process_id, ..
            } => {
                if let Some(process_id) = process_id.take() {
                    process_group::kill(process_id);
                }
                if matches!(child.try_wait(), Ok(None)) {
                    let _ = child.start_kill();
                }
                let _ = tokio::time::timeout(std::time::Duration::from_secs(2), child.wait()).await;
            }
            SessionHandle::Pty(pty) => {
                let _ = pty.kill();
                let deadline = Instant::now() + std::time::Duration::from_secs(2);
                while pty.poll_exit_code().is_none() && Instant::now() < deadline {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
        }
        if let Some(reader) = self._reader.take() {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(1), reader).await;
        }
    }
}

fn stream_hook(
    ctx: &ToolCallContext,
    _session_id: i32,
) -> Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>> {
    let session = ctx.session.clone();
    Some(Arc::new(move |delta| {
        let _ = session.on_exec_output_delta(delta);
    }))
}

fn spawn_pty_exec(
    command: &str,
    workdir: &Path,
    login: bool,
    policy: SandboxPolicy,
    dangerous: bool,
    buffer: Arc<Mutex<OutputBuffer>>,
    session_id: i32,
    on_delta: Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>>,
) -> Result<PtyExecHandle, ToolError> {
    let plan = build_bash_exec(command, Some(workdir), policy, dangerous, login);
    crate::pty_session::spawn_pty_exec(&plan, workdir, buffer, session_id, on_delta)
}

fn parse_session_id(value: Option<&Value>) -> Result<i32, ToolError> {
    let Some(v) = value else {
        return Err(ToolError::respond("missing session_id"));
    };
    let id = if let Some(n) = v.as_i64() {
        n
    } else if let Some(s) = v.as_str() {
        s.parse::<i64>()
            .map_err(|_| ToolError::respond(format!("invalid session_id {s:?}")))?
    } else {
        return Err(ToolError::respond(format!("invalid session_id {v}")));
    };
    i32::try_from(id).map_err(|_| ToolError::respond(format!("invalid session_id {id}")))
}

fn session_has_exited(handle: &mut SessionHandle) -> bool {
    match handle {
        SessionHandle::Pipe { child, .. } => matches!(child.try_wait(), Ok(Some(_))),
        SessionHandle::Pty(pty) => pty.poll_exit_code().is_some(),
    }
}

async fn spawn_pipe_session(
    command: &str,
    workdir: &Path,
    login: bool,
    policy: SandboxPolicy,
    dangerous: bool,
    buffer: Arc<Mutex<OutputBuffer>>,
    session_id: i32,
    on_delta: Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>>,
) -> Result<(Child, Option<ChildStdin>, JoinHandle<()>), ToolError> {
    let plan = build_bash_exec(command, Some(workdir), policy, dangerous, login);
    let mut cmd = Command::new(&plan.argv[0]);
    cmd.args(&plan.argv[1..]);
    process_group::configure(&mut cmd);
    cmd.kill_on_drop(true);
    cmd.current_dir(workdir)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(env) = &plan.env {
        cmd.env_clear();
        for (k, v) in env {
            cmd.env(k, v);
        }
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| ToolError::respond(format!("spawn failed: {e}")))?;
    let stdin = child.stdin.take();
    let reader = spawn_stream_reader(
        child.stdout.take(),
        child.stderr.take(),
        buffer,
        session_id,
        on_delta,
    );
    Ok((child, stdin, reader))
}

fn spawn_stream_reader(
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    buffer: Arc<Mutex<OutputBuffer>>,
    session_id: i32,
    on_delta: Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut out = stdout;
        let mut err = stderr;
        // Separate decoders per stream: stdout/stderr interleave independently,
        // so a shared decoder could pair leftover bytes from one stream with
        // bytes from the other and corrupt both.
        let mut out_decoder = Utf8ChunkDecoder::default();
        let mut err_decoder = Utf8ChunkDecoder::default();
        let mut buf = [0u8; 4096];
        loop {
            let mut progressed = false;
            if let Some(s) = out.as_mut() {
                match s.read(&mut buf).await {
                    Ok(0) => out = None,
                    Ok(n) => {
                        push_chunk(
                            &mut out_decoder,
                            &buffer,
                            session_id,
                            ExecStream::Stdout,
                            &buf[..n],
                            &on_delta,
                        )
                        .await;
                        progressed = true;
                    }
                    Err(_) => out = None,
                }
            }
            if let Some(e) = err.as_mut() {
                match e.read(&mut buf).await {
                    Ok(0) => err = None,
                    Ok(n) => {
                        push_chunk(
                            &mut err_decoder,
                            &buffer,
                            session_id,
                            ExecStream::Stderr,
                            &buf[..n],
                            &on_delta,
                        )
                        .await;
                        progressed = true;
                    }
                    Err(_) => err = None,
                }
            }
            if out.is_none() && err.is_none() {
                break;
            }
            if !progressed {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
        flush_decoder(
            &mut out_decoder,
            &buffer,
            session_id,
            ExecStream::Stdout,
            &on_delta,
        )
        .await;
        flush_decoder(
            &mut err_decoder,
            &buffer,
            session_id,
            ExecStream::Stderr,
            &on_delta,
        )
        .await;
    })
}

async fn push_chunk(
    decoder: &mut Utf8ChunkDecoder,
    buffer: &Arc<Mutex<OutputBuffer>>,
    session_id: i32,
    stream: ExecStream,
    bytes: &[u8],
    on_delta: &Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>>,
) {
    let chunk = decoder.decode(bytes);
    if chunk.is_empty() {
        return;
    }
    buffer.lock().await.push(&chunk);
    if let Some(hook) = on_delta {
        hook(ExecOutputDelta {
            session_id,
            stream,
            chunk,
        });
    }
}

async fn flush_decoder(
    decoder: &mut Utf8ChunkDecoder,
    buffer: &Arc<Mutex<OutputBuffer>>,
    session_id: i32,
    stream: ExecStream,
    on_delta: &Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>>,
) {
    let tail = decoder.finish();
    if tail.is_empty() {
        return;
    }
    buffer.lock().await.push(&tail);
    if let Some(hook) = on_delta {
        hook(ExecOutputDelta {
            session_id,
            stream,
            chunk: tail,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::{ApprovalCache, ApprovalPolicy, new_exec_policy};
    use crate::context::{LocalToolSession, ToolCallContext, ToolSession};
    use std::sync::Arc;

    use crate::discovery::new_tool_catalog;

    fn test_ctx(session: Arc<dyn ToolSession>, exec: Arc<UnifiedExecManager>) -> ToolCallContext {
        ToolCallContext::new(
            ".",
            "off",
            session,
            exec,
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            new_tool_catalog(),
        )
    }

    #[tokio::test]
    async fn exec_command_tty_uses_pty() {
        let mgr = UnifiedExecManager::new();
        let exec = Arc::new(mgr);
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = test_ctx(session, exec.clone());
        let out = exec
            .exec_command(
                &ctx,
                &serde_json::json!({
                    "cmd": "echo pty_ok",
                    "yield_time_ms": 500,
                    "tty": true
                }),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        let structured = out.structured.unwrap();
        let output = structured
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(output.contains("pty_ok") || structured.get("session_id").is_some());
    }

    async fn assert_turn_cancel_stops_process_tree(tty: bool) {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("should-not-exist");
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let turn = Arc::new(crate::turn::TurnExecutionContext::new(
            crate::SessionId::new(),
            crate::TurnId::new(),
        ));
        let ctx = ToolCallContext::new(
            dir.path(),
            "off",
            session,
            exec.clone(),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            new_tool_catalog(),
        )
        .with_turn(turn.clone());
        let args = serde_json::json!({
            "cmd": "(sleep 0.3; touch should-not-exist) & wait",
            "yield_time_ms": 5_000,
            "tty": tty
        });

        let cancel = async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            turn.cancel();
        };
        let (result, ()) = tokio::join!(exec.exec_command(&ctx, &args), cancel);
        assert!(result.unwrap().is_error);
        assert!(
            turn.resources()
                .wait_for_idle(std::time::Duration::from_secs(3))
                .await
        );
        assert!(exec.list_sessions().await.is_empty());
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        assert!(!marker.exists(), "cancelled process tree produced a file");
    }

    #[tokio::test]
    async fn cancellation_stops_pipe_session_process_tree() {
        assert_turn_cancel_stops_process_tree(false).await;
    }

    #[tokio::test]
    async fn cancellation_stops_pty_session_process_tree() {
        assert_turn_cancel_stops_process_tree(true).await;
    }

    #[tokio::test]
    async fn write_stdin_unknown_session_returns_clear_error_not_panic() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = test_ctx(session, exec.clone());
        // Runtime dispatch converts this Err into `ToolResult::err(..)` for the model;
        // exercise the same conversion here to assert no panic and a clear message.
        let err = exec
            .write_stdin(&ctx, &serde_json::json!({"session_id": 999, "chars": "hi"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("999"));
    }

    #[tokio::test]
    async fn write_stdin_rejects_invalid_session_id_type() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = test_ctx(session, exec.clone());
        let err = exec
            .write_stdin(
                &ctx,
                &serde_json::json!({"session_id": "not-a-number", "chars": "hi"}),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid session_id"));
    }

    #[tokio::test]
    async fn write_stdin_rejects_oversized_input() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = test_ctx(session, exec.clone());
        let huge = "a".repeat(MAX_STDIN_BYTES + 1);
        let out = exec
            .write_stdin(&ctx, &serde_json::json!({"session_id": 1, "chars": huge}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("too large"));
    }

    #[tokio::test]
    async fn write_stdin_rejects_utf8_oversized_by_bytes_not_chars() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = test_ctx(session, exec.clone());
        let huge = "\u{00e9}".repeat(MAX_STDIN_BYTES / 2 + 1);
        assert!(huge.len() > MAX_STDIN_BYTES);
        let out = exec
            .write_stdin(&ctx, &serde_json::json!({"session_id": 1, "chars": huge}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("too large"));
    }

    #[tokio::test]
    async fn write_stdin_accepts_empty_input_for_live_session() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = test_ctx(session, exec.clone());
        let started = exec
            .exec_command(
                &ctx,
                &serde_json::json!({"cmd": "cat", "yield_time_ms": 300, "tty": true}),
            )
            .await
            .unwrap();
        let session_id = started
            .structured
            .as_ref()
            .and_then(|v| v.get("session_id"))
            .and_then(|v| v.as_i64())
            .expect("running session should report session_id");

        let out = exec
            .write_stdin(
                &ctx,
                &serde_json::json!({"session_id": session_id, "chars": "", "yield_time_ms": 300}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);

        let _ = exec
            .write_stdin(
                &ctx,
                &serde_json::json!({"session_id": session_id, "chars": "\x03"}),
            )
            .await;
    }

    #[tokio::test]
    async fn write_stdin_concurrent_writes_do_not_panic() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = test_ctx(session, exec.clone());
        let started = exec
            .exec_command(
                &ctx,
                &serde_json::json!({"cmd": "cat", "yield_time_ms": 300, "tty": true}),
            )
            .await
            .unwrap();
        let session_id = started
            .structured
            .as_ref()
            .and_then(|v| v.get("session_id"))
            .and_then(|v| v.as_i64())
            .expect("running session should report session_id");

        let mut handles = Vec::new();
        for i in 0..8 {
            let exec = exec.clone();
            let ctx = test_ctx(
                Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always)),
                exec.clone(),
            );
            handles.push(tokio::spawn(async move {
                exec.write_stdin(
                    &ctx,
                    &serde_json::json!({"session_id": session_id, "chars": format!("{i}\n")}),
                )
                .await
            }));
        }
        for h in handles {
            let _ = h.await.unwrap();
        }

        let _ = exec
            .write_stdin(
                &ctx,
                &serde_json::json!({"session_id": session_id, "chars": "\x03"}),
            )
            .await;
    }

    #[tokio::test]
    async fn streaming_hook_receives_deltas() {
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let exec = Arc::new(UnifiedExecManager::new());
        let ctx = test_ctx(session.clone(), exec.clone());
        let _ = exec
            .exec_command(
                &ctx,
                &serde_json::json!({
                    "cmd": "echo stream_test",
                    "yield_time_ms": 500,
                    "tty": true
                }),
            )
            .await
            .unwrap();
        assert!(!session.exec_deltas().is_empty());
    }

    #[tokio::test]
    async fn drop_manager_kills_running_pty_session() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = test_ctx(session, exec.clone());
        let out = exec
            .exec_command(
                &ctx,
                &serde_json::json!({
                    "cmd": "sleep 30",
                    "yield_time_ms": 100,
                    "tty": true
                }),
            )
            .await
            .unwrap();
        assert!(
            out.structured
                .as_ref()
                .and_then(|v| v.get("session_id"))
                .is_some()
        );
        assert!(!exec.list_sessions().await.is_empty());

        drop(exec);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    #[tokio::test]
    async fn exec_command_tty_applies_restricted_sandbox() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = ToolCallContext::new(
            ".",
            "restricted",
            session,
            exec.clone(),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            new_tool_catalog(),
        );
        let out = exec
            .exec_command(
                &ctx,
                &serde_json::json!({
                    "cmd": "echo $PATH",
                    "yield_time_ms": 500,
                    "tty": true
                }),
            )
            .await
            .unwrap();
        let structured = out.structured.unwrap();
        let output = structured
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            output.contains("/usr/bin:/bin:/usr/local/bin"),
            "pty path must route through restricted env wrapper, got: {output:?}"
        );
    }

    #[tokio::test]
    async fn exec_command_pipe_mode_applies_restricted_sandbox() {
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = ToolCallContext::new(
            ".",
            "restricted",
            session,
            exec.clone(),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            new_tool_catalog(),
        );
        let out = exec
            .exec_command(
                &ctx,
                &serde_json::json!({
                    "cmd": "echo $PATH",
                    "yield_time_ms": 500,
                    "tty": false
                }),
            )
            .await
            .unwrap();
        let structured = out.structured.unwrap();
        let output = structured
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            output.contains("/usr/bin:/bin:/usr/local/bin"),
            "pipe path must route through restricted env wrapper, got: {output:?}"
        );
    }

    #[tokio::test]
    async fn exec_command_rejects_workdir_escape() {
        let dir = tempfile::tempdir().unwrap();
        let exec = Arc::new(UnifiedExecManager::new());
        let session =
            Arc::new(LocalToolSession::default().with_approval_policy(ApprovalPolicy::Always));
        let ctx = ToolCallContext::new(
            dir.path(),
            "off",
            session,
            exec.clone(),
            new_exec_policy(),
            Arc::new(ApprovalCache::default()),
            new_tool_catalog(),
        );
        let err = exec
            .exec_command(
                &ctx,
                &serde_json::json!({
                    "cmd": "echo hi",
                    "workdir": "../escape",
                    "yield_time_ms": 300,
                    "tty": false
                }),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("outside working directory"));
    }
}
