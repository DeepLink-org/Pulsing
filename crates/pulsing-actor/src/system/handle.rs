//! Actor handle and statistics types

use crate::actor::{ActorId, ActorPath, Envelope};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Actor runtime statistics
#[derive(Debug, Default)]
pub struct ActorStats {
    /// Number of times the actor started
    pub start_count: AtomicU64,
    /// Number of times the actor stopped
    pub stop_count: AtomicU64,
    /// Number of messages processed
    pub message_count: AtomicU64,
}

impl ActorStats {
    /// Increment stop count
    pub fn inc_stop(&self) {
        self.stop_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment message count
    pub fn inc_message(&self) {
        self.message_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Convert to JSON representation
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "start_count": self.start_count.load(Ordering::Relaxed),
            "stop_count": self.stop_count.load(Ordering::Relaxed),
            "message_count": self.message_count.load(Ordering::Relaxed),
        })
    }
}

/// Local actor handle - internal representation of a running actor
pub(crate) struct LocalActorHandle {
    /// Sender to the actor's mailbox
    pub sender: mpsc::Sender<Envelope>,

    /// Actor task handle
    pub join_handle: JoinHandle<()>,

    /// Cancellation token for graceful shutdown of this specific actor
    pub cancel_token: CancellationToken,

    /// Runtime statistics
    pub stats: Arc<ActorStats>,

    /// Static metadata provided by the actor
    pub metadata: HashMap<String, String>,

    /// Named actor path (if this is a named actor)
    pub named_path: Option<ActorPath>,

    /// Full actor ID
    pub actor_id: ActorId,
}
