//! Tokio runtime + standalone ``ActorSystem`` for the pulsing-cli binary.

use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, Result};
use pulsing_actor::prelude::{ActorSystem, SystemConfig};
use tokio::runtime::Handle;

use crate::interop::{drain_vm_queue, schedule_vm_work};

struct CliRuntime {
    _thread: JoinHandle<()>,
    system: Arc<ActorSystem>,
    handle: Handle,
}

static RUNTIME: OnceLock<CliRuntime> = OnceLock::new();

/// Start (once) a standalone actor system for the CLI process.
pub fn ensure_runtime() -> Result<Arc<ActorSystem>> {
    Ok(Arc::clone(&runtime()?.system))
}

pub fn tokio_handle() -> Result<Handle> {
    Ok(runtime()?.handle.clone())
}

/// Poll the RustPython VM work queue from the tokio runtime thread.
pub fn start_vm_drainer() {
    use std::sync::atomic::{AtomicBool, Ordering};

    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let handle = tokio_handle().expect("tokio runtime");
    handle.spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(1)).await;
            schedule_vm_work(Box::new(drain_vm_queue));
        }
    });
}

fn runtime() -> Result<&'static CliRuntime> {
    if let Some(rt) = RUNTIME.get() {
        return Ok(rt);
    }

    let (tx, rx) = std::sync::mpsc::sync_channel::<(Arc<ActorSystem>, Handle)>(1);
    let thread = thread::Builder::new()
        .name("pulsing-actor-runtime".into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("pulsing-cli-tokio")
                .build()
                .expect("failed to build tokio runtime");
            let handle = runtime.handle().clone();
            let system = runtime
                .block_on(async { ActorSystem::new(SystemConfig::standalone()).await })
                .expect("ActorSystem::new failed");
            let _ = tx.send((Arc::clone(&system), handle.clone()));
            runtime.block_on(async {
                loop {
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                }
            });
        })
        .context("failed to spawn actor runtime thread")?;

    let (system, handle) = rx
        .recv()
        .context("actor runtime thread exited before ActorSystem was ready")?;
    let _ = RUNTIME.set(CliRuntime {
        _thread: thread,
        system,
        handle,
    });
    RUNTIME.get().context("runtime init race")
}
