//! Load balancing utilities for the actor system
//!
//! This module provides per-node load tracking and worker adapters for
//! integrating with the load balancing policies.

use crate::cluster::MemberInfo;
use crate::cluster::MemberStatus;
use crate::policies::Worker;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Per-node load tracking with activity timestamp for cleanup
///
/// Tracks the current load (in-flight requests) and total processed requests
/// for a remote node. Includes timestamp tracking for stale entry cleanup.
#[derive(Debug)]
pub struct NodeLoadTracker {
    /// Current in-flight requests to this node
    load: AtomicUsize,
    /// Total requests processed
    processed: AtomicU64,
    /// Last activity timestamp (Unix millis) for stale entry cleanup
    last_activity_millis: AtomicU64,
}

impl Default for NodeLoadTracker {
    fn default() -> Self {
        Self {
            load: AtomicUsize::new(0),
            processed: AtomicU64::new(0),
            last_activity_millis: AtomicU64::new(Self::current_millis()),
        }
    }
}

impl NodeLoadTracker {
    /// Create a new load tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Get current timestamp in milliseconds
    fn current_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Update the last activity timestamp
    fn touch(&self) {
        self.last_activity_millis
            .store(Self::current_millis(), Ordering::Relaxed);
    }

    /// Get current load (in-flight requests)
    pub fn load(&self) -> usize {
        self.load.load(Ordering::Relaxed)
    }

    /// Increment the load counter
    pub fn increment(&self) {
        self.load.fetch_add(1, Ordering::Relaxed);
        self.touch();
    }

    /// Decrement the load counter
    pub fn decrement(&self) {
        self.load.fetch_sub(1, Ordering::Relaxed);
        self.touch();
    }

    /// Get total processed requests
    pub fn processed(&self) -> u64 {
        self.processed.load(Ordering::Relaxed)
    }

    /// Increment the processed counter
    pub fn increment_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
        self.touch();
    }

    /// Returns elapsed time since last activity
    pub fn last_activity_elapsed(&self) -> Duration {
        let last = self.last_activity_millis.load(Ordering::Relaxed);
        let now = Self::current_millis();
        Duration::from_millis(now.saturating_sub(last))
    }

    /// Returns true if this tracker has been inactive for longer than the threshold
    pub fn is_stale(&self, threshold: Duration) -> bool {
        self.last_activity_elapsed() > threshold
    }
}

/// Wrapper to adapt MemberInfo to the Worker trait for load balancing
///
/// This allows cluster member information to be used with the load balancing
/// policies defined in the `policies` module.
#[derive(Debug)]
pub struct MemberWorker {
    url: String,
    is_alive: bool,
    /// Shared load tracker for this node
    load_tracker: Arc<NodeLoadTracker>,
}

impl MemberWorker {
    /// Create a new MemberWorker from a MemberInfo
    pub fn new(member: &MemberInfo, load_tracker: Arc<NodeLoadTracker>) -> Self {
        Self {
            url: member.addr.to_string(),
            is_alive: member.status == MemberStatus::Alive,
            load_tracker,
        }
    }
}

impl Worker for MemberWorker {
    fn url(&self) -> &str {
        &self.url
    }

    fn is_healthy(&self) -> bool {
        self.is_alive
    }

    fn set_healthy(&mut self, healthy: bool) {
        self.is_alive = healthy;
    }

    fn load(&self) -> usize {
        self.load_tracker.load()
    }

    fn increment_load(&self) {
        self.load_tracker.increment();
    }

    fn decrement_load(&self) {
        self.load_tracker.decrement();
    }

    fn increment_processed(&self) {
        self.load_tracker.increment_processed();
    }

    fn processed(&self) -> u64 {
        self.load_tracker.processed()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_load_tracker_default() {
        let tracker = NodeLoadTracker::new();
        assert_eq!(tracker.load(), 0);
        assert_eq!(tracker.processed(), 0);
    }

    #[test]
    fn test_node_load_tracker_increment_decrement() {
        let tracker = NodeLoadTracker::new();

        tracker.increment();
        tracker.increment();
        assert_eq!(tracker.load(), 2);

        tracker.decrement();
        assert_eq!(tracker.load(), 1);
    }

    #[test]
    fn test_node_load_tracker_processed() {
        let tracker = NodeLoadTracker::new();

        tracker.increment_processed();
        tracker.increment_processed();
        tracker.increment_processed();
        assert_eq!(tracker.processed(), 3);
    }

    #[test]
    fn test_node_load_tracker_staleness() {
        let tracker = NodeLoadTracker::new();

        // Just created, should not be stale
        assert!(!tracker.is_stale(Duration::from_secs(1)));

        // Elapsed time should be very small
        let elapsed = tracker.last_activity_elapsed();
        assert!(elapsed < Duration::from_secs(1));
    }

    #[test]
    fn test_member_worker() {
        let addr: std::net::SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let member = MemberInfo {
            node_id: crate::actor::NodeId::new(1),
            addr,
            gossip_addr: addr,
            status: MemberStatus::Alive,
            incarnation: 0,
            last_update: None,
        };
        let tracker = Arc::new(NodeLoadTracker::new());
        let worker = MemberWorker::new(&member, tracker);

        assert!(worker.is_healthy());
        assert_eq!(worker.url(), "127.0.0.1:8000");
        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_member_worker_dead_node() {
        let addr: std::net::SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let member = MemberInfo {
            node_id: crate::actor::NodeId::new(1),
            addr,
            gossip_addr: addr,
            status: MemberStatus::Dead,
            incarnation: 0,
            last_update: None,
        };
        let tracker = Arc::new(NodeLoadTracker::new());
        let worker = MemberWorker::new(&member, tracker);

        assert!(!worker.is_healthy());
    }
}
