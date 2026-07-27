//! PTY-backed exec session (portable-pty).

use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use portable_pty::{ChildKiller, CommandBuilder, PtySize, native_pty_system};

use crate::error::ToolError;
use crate::exec_output::{
    ExecOutputDelta, ExecStream, OutputBuffer, RUNNING_EXIT_SENTINEL, Utf8ChunkDecoder,
};
use crate::process_group;
use crate::sandbox::BashExecPlan;

enum PtyCommand {
    Write(Vec<u8>),
    Kill,
}

/// Handle to a background PTY session thread.
pub struct PtyExecHandle {
    cmd_tx: mpsc::Sender<PtyCommand>,
    killer: Arc<Mutex<Box<dyn ChildKiller + Send + Sync>>>,
    process_id: Option<u32>,
    pub exit_code: Arc<AtomicI32>,
    _thread: JoinHandle<()>,
}

impl Drop for PtyExecHandle {
    fn drop(&mut self) {
        if self.poll_exit_code().is_none() {
            let _ = self.kill();
        }
    }
}

impl PtyExecHandle {
    pub fn write_stdin(&self, data: &[u8]) -> Result<(), ToolError> {
        self.cmd_tx
            .send(PtyCommand::Write(data.to_vec()))
            .map_err(|e| ToolError::respond(format!("pty stdin channel closed: {e}")))
    }

    pub fn kill(&self) -> Result<(), ToolError> {
        if let Some(process_id) = self.process_id {
            process_group::kill(process_id);
        }
        self.killer
            .lock()
            .map_err(|_| ToolError::respond("pty killer lock poisoned"))?
            .kill()
            .map_err(|e| ToolError::respond(format!("pty kill failed: {e}")))?;
        // Keep the command for implementations where killing does not
        // immediately wake the PTY reader loop.
        let _ = self.cmd_tx.send(PtyCommand::Kill);
        Ok(())
    }

    pub fn poll_exit_code(&self) -> Option<i32> {
        let code = self.exit_code.load(Ordering::SeqCst);
        if code == RUNNING_EXIT_SENTINEL {
            None
        } else {
            Some(code)
        }
    }
}

pub fn spawn_pty_exec(
    plan: &BashExecPlan,
    workdir: &Path,
    buffer: Arc<tokio::sync::Mutex<OutputBuffer>>,
    session_id: i32,
    on_delta: Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>>,
) -> Result<PtyExecHandle, ToolError> {
    let cmd_tx_out;
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (killer_tx, killer_rx) = mpsc::sync_channel(1);
    cmd_tx_out = cmd_tx;
    let exit_code = Arc::new(AtomicI32::new(RUNNING_EXIT_SENTINEL));
    let exit_out = exit_code.clone();
    let exit_guard = exit_code.clone();
    let plan = plan.clone();
    let wd = workdir.to_path_buf();

    let thread = thread::spawn(move || {
        if let Err(e) = run_pty_thread(
            &plan, &wd, cmd_rx, killer_tx, buffer, session_id, on_delta, exit_out,
        ) {
            tracing::warn!("pty session ended with error: {e}");
        }
        // Setup can fail before the wait loop ever runs (e.g. openpty/spawn
        // errors above); without this the session would sit in the table
        // forever reporting "still running" since nothing else advances
        // `exit_code` on that path.
        if exit_guard.load(Ordering::SeqCst) == RUNNING_EXIT_SENTINEL {
            exit_guard.store(-1, Ordering::SeqCst);
        }
    });
    let (killer, process_id) = killer_rx
        .recv()
        .map_err(|_| ToolError::respond("pty process failed before exposing its kill handle"))?;

    Ok(PtyExecHandle {
        cmd_tx: cmd_tx_out,
        killer: Arc::new(Mutex::new(killer)),
        process_id,
        exit_code,
        _thread: thread,
    })
}

fn run_pty_thread(
    plan: &BashExecPlan,
    workdir: &Path,
    cmd_rx: mpsc::Receiver<PtyCommand>,
    killer_tx: mpsc::SyncSender<(Box<dyn ChildKiller + Send + Sync>, Option<u32>)>,
    buffer: Arc<tokio::sync::Mutex<OutputBuffer>>,
    session_id: i32,
    on_delta: Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>>,
    exit_code: Arc<AtomicI32>,
) -> Result<(), ToolError> {
    // Surface setup failures (openpty/spawn/reader/writer) in the session's
    // own output buffer, not just tracing logs — otherwise callers only see
    // an empty transcript with no clue why the command never produced output.
    let report_err = |msg: String| -> ToolError {
        if let Ok(mut b) = buffer.try_lock() {
            b.push(&format!("{msg}\n"));
        }
        ToolError::respond(msg)
    };

    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: 24,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| report_err(format!("openpty failed: {e}")))?;

    let mut cmd_builder = CommandBuilder::new(&plan.argv[0]);
    for arg in &plan.argv[1..] {
        cmd_builder.arg(arg);
    }
    cmd_builder.cwd(workdir);
    if let Some(env) = &plan.env {
        cmd_builder.env_clear();
        for (k, v) in env {
            cmd_builder.env(k, v);
        }
    }

    let mut child = pair
        .slave
        .spawn_command(cmd_builder)
        .map_err(|e| report_err(format!("pty spawn failed: {e}")))?;
    killer_tx
        .send((child.clone_killer(), child.process_id()))
        .map_err(|_| report_err("pty owner dropped during startup".into()))?;

    let mut reader = pair
        .master
        .try_clone_reader()
        .map_err(|e| report_err(format!("pty reader failed: {e}")))?;
    let mut writer = pair
        .master
        .take_writer()
        .map_err(|e| report_err(format!("pty writer failed: {e}")))?;

    let mut decoder = Utf8ChunkDecoder::default();
    let mut buf = [0u8; 4096];
    loop {
        loop {
            match cmd_rx.try_recv() {
                Ok(PtyCommand::Write(data)) => {
                    if let Err(e) = writer.write_all(&data) {
                        if let Ok(mut b) = buffer.try_lock() {
                            b.push(&format!("write stdin failed: {e}\n"));
                        }
                    } else {
                        let _ = writer.flush();
                    }
                }
                Ok(PtyCommand::Kill) => {
                    let _ = child.kill();
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    let _ = child.kill();
                    break;
                }
            }
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                exit_code.store(status.exit_code() as i32, Ordering::SeqCst);
                break;
            }
            Ok(None) => {}
            Err(e) => {
                exit_code.store(-1, Ordering::SeqCst);
                return Err(ToolError::respond(format!("pty wait failed: {e}")));
            }
        }

        match reader.read(&mut buf) {
            Ok(0) => {
                thread::sleep(std::time::Duration::from_millis(10));
            }
            Ok(n) => {
                emit_chunk(&mut decoder, &buf[..n], &buffer, session_id, &on_delta);
            }
            Err(_) => thread::sleep(std::time::Duration::from_millis(10)),
        }
    }

    // Drain remaining output.
    while let Ok(n) = reader.read(&mut buf) {
        if n == 0 {
            break;
        }
        emit_chunk(&mut decoder, &buf[..n], &buffer, session_id, &on_delta);
    }
    let tail = decoder.finish();
    if !tail.is_empty() {
        if let Ok(mut b) = buffer.try_lock() {
            b.push(&tail);
        }
        if let Some(hook) = &on_delta {
            hook(ExecOutputDelta {
                session_id,
                stream: ExecStream::Pty,
                chunk: tail,
            });
        }
    }

    Ok(())
}

fn emit_chunk(
    decoder: &mut Utf8ChunkDecoder,
    bytes: &[u8],
    buffer: &Arc<tokio::sync::Mutex<OutputBuffer>>,
    session_id: i32,
    on_delta: &Option<Arc<dyn Fn(ExecOutputDelta) + Send + Sync>>,
) {
    let chunk = decoder.decode(bytes);
    if chunk.is_empty() {
        return;
    }
    if let Ok(mut b) = buffer.try_lock() {
        b.push(&chunk);
    }
    if let Some(hook) = on_delta {
        hook(ExecOutputDelta {
            session_id,
            stream: ExecStream::Pty,
            chunk,
        });
    }
}
