//! Turn-scoped cancellation and resource ownership.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::protocol::{SessionId, TurnId};

type CancelResource = Arc<dyn Fn() + Send + Sync>;

struct ResourceEntry {
    kind: String,
    cancel: CancelResource,
}

/// Registry of external resources owned by one Forge turn.
///
/// A resource stays registered until its guard is dropped. Cancellation first
/// invokes every registered resource's non-blocking canceller; terminal turn
/// events may then wait until all guards have been released.
pub struct TurnResourceRegistry {
    next_id: AtomicU64,
    entries: Mutex<HashMap<u64, ResourceEntry>>,
    idle: Notify,
}

impl Default for TurnResourceRegistry {
    fn default() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            entries: Mutex::new(HashMap::new()),
            idle: Notify::new(),
        }
    }
}

impl TurnResourceRegistry {
    pub fn register(
        self: &Arc<Self>,
        kind: impl Into<String>,
        cancel: impl Fn() + Send + Sync + 'static,
    ) -> TurnResourceGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.entries.lock().expect("turn resources").insert(
            id,
            ResourceEntry {
                kind: kind.into(),
                cancel: Arc::new(cancel),
            },
        );
        TurnResourceGuard {
            registry: Arc::downgrade(self),
            id,
        }
    }

    pub fn register_passive(self: &Arc<Self>, kind: impl Into<String>) -> TurnResourceGuard {
        self.register(kind, || {})
    }

    pub fn cancel_all(&self) {
        let cancellers = self
            .entries
            .lock()
            .expect("turn resources")
            .values()
            .map(|entry| entry.cancel.clone())
            .collect::<Vec<_>>();
        for cancel in cancellers {
            cancel();
        }
    }

    pub fn active_count(&self) -> usize {
        self.entries.lock().expect("turn resources").len()
    }

    pub fn active_kinds(&self) -> Vec<String> {
        self.entries
            .lock()
            .expect("turn resources")
            .values()
            .map(|entry| entry.kind.clone())
            .collect()
    }

    pub async fn wait_for_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            let notified = self.idle.notified();
            if self.active_count() == 0 {
                return true;
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            if tokio::time::timeout(remaining, notified).await.is_err() {
                return self.active_count() == 0;
            }
        }
    }

    pub async fn wait_for_idle_unbounded(&self) {
        loop {
            let notified = self.idle.notified();
            if self.active_count() == 0 {
                return;
            }
            notified.await;
        }
    }

    fn release(&self, id: u64) {
        if self
            .entries
            .lock()
            .expect("turn resources")
            .remove(&id)
            .is_some()
        {
            self.idle.notify_waiters();
        }
    }
}

/// RAII ownership token for one external turn resource.
pub struct TurnResourceGuard {
    registry: Weak<TurnResourceRegistry>,
    id: u64,
}

impl Drop for TurnResourceGuard {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            registry.release(self.id);
        }
    }
}

/// Execution identity, cancellation signal, and resource registry for one turn.
pub struct TurnExecutionContext {
    pub session_id: SessionId,
    pub turn_id: TurnId,
    cancellation: CancellationToken,
    resources: Arc<TurnResourceRegistry>,
}

impl TurnExecutionContext {
    pub fn new(session_id: SessionId, turn_id: TurnId) -> Self {
        Self::with_cancellation(session_id, turn_id, CancellationToken::new())
    }

    pub fn with_cancellation(
        session_id: SessionId,
        turn_id: TurnId,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            session_id,
            turn_id,
            cancellation,
            resources: Arc::new(TurnResourceRegistry::default()),
        }
    }

    pub fn cancellation(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    pub fn resources(&self) -> &Arc<TurnResourceRegistry> {
        &self.resources
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
        self.resources.cancel_all();
    }

    pub async fn cancel_and_wait(&self, timeout: Duration) -> bool {
        self.cancel();
        self.resources.wait_for_idle(timeout).await
    }

    pub async fn cleanup_and_wait(&self, timeout: Duration) -> bool {
        self.resources.cancel_all();
        self.resources.wait_for_idle(timeout).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[tokio::test]
    async fn cancellation_invokes_resources_and_waits_for_release() {
        let turn = TurnExecutionContext::new(SessionId::new(), TurnId::new());
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = cancelled.clone();
        let guard = turn
            .resources()
            .register("test", move || flag.store(true, Ordering::SeqCst));

        turn.cancel();
        assert!(cancelled.load(Ordering::SeqCst));
        assert_eq!(turn.resources().active_kinds(), vec!["test"]);
        assert!(
            !turn
                .resources()
                .wait_for_idle(Duration::from_millis(10))
                .await
        );
        drop(guard);
        assert!(
            turn.resources()
                .wait_for_idle(Duration::from_millis(10))
                .await
        );
    }
}
