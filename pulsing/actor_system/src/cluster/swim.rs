//! SWIM failure detection protocol
//!
//! Implements the SWIM (Scalable Weakly-consistent Infection-style Membership) protocol
//! for detecting failed nodes in the cluster.

use crate::actor::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// SWIM configuration
#[derive(Clone, Debug)]
pub struct SwimConfig {
    /// Ping interval
    pub ping_interval: Duration,

    /// Ping timeout before indirect probe
    pub ping_timeout: Duration,

    /// Number of indirect probes
    pub indirect_probes: usize,

    /// Suspicion timeout before marking dead
    pub suspicion_timeout: Duration,
}

impl Default for SwimConfig {
    fn default() -> Self {
        Self {
            ping_interval: Duration::from_millis(500),
            ping_timeout: Duration::from_millis(200),
            indirect_probes: 3,
            suspicion_timeout: Duration::from_secs(5),
        }
    }
}

/// SWIM protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwimMessage {
    /// Direct ping
    Ping { seq: u64, from: NodeId },

    /// Ping acknowledgment
    Ack { seq: u64, from: NodeId },

    /// Indirect ping request
    PingReq {
        seq: u64,
        from: NodeId,
        target: NodeId,
        target_addr: SocketAddr,
    },

    /// Indirect ping acknowledgment
    PingReqAck {
        seq: u64,
        from: NodeId,
        target: NodeId,
    },
}

/// Pending ping state
struct PendingPing {
    target: NodeId,
    sent_at: Instant,
}

/// SWIM failure detector
pub struct SwimDetector {
    local_node: NodeId,
    config: SwimConfig,
    seq: AtomicU64,
    pending_pings: RwLock<HashMap<u64, PendingPing>>,
}

impl Clone for SwimDetector {
    fn clone(&self) -> Self {
        Self {
            local_node: self.local_node.clone(),
            config: self.config.clone(),
            seq: AtomicU64::new(self.seq.load(Ordering::SeqCst)),
            pending_pings: RwLock::new(HashMap::new()),
        }
    }
}

impl SwimDetector {
    /// Create a new SWIM detector
    pub fn new(local_node: NodeId, config: SwimConfig) -> Self {
        Self {
            local_node,
            config,
            seq: AtomicU64::new(0),
            pending_pings: RwLock::new(HashMap::new()),
        }
    }

    /// Get ping interval
    pub fn ping_interval(&self) -> Duration {
        self.config.ping_interval
    }

    /// Create a new ping message
    pub fn create_ping(&self) -> (u64, SwimMessage) {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let ping = SwimMessage::Ping {
            seq,
            from: self.local_node.clone(),
        };
        (seq, ping)
    }

    /// Create an ack message
    pub fn create_ack(&self, seq: u64) -> SwimMessage {
        SwimMessage::Ack {
            seq,
            from: self.local_node.clone(),
        }
    }

    /// Record that a ping was sent
    pub async fn ping_sent(&self, seq: u64, target: NodeId) {
        let mut pending = self.pending_pings.write().await;
        pending.insert(
            seq,
            PendingPing {
                target,
                sent_at: Instant::now(),
            },
        );
    }

    /// Record that an ack was received
    pub async fn ack_received(&self, seq: u64) {
        let mut pending = self.pending_pings.write().await;
        pending.remove(&seq);
    }

    /// Check for ping timeouts
    /// Returns (node_id, should_suspect)
    pub async fn check_timeouts(&self) -> Vec<(NodeId, bool)> {
        let mut pending = self.pending_pings.write().await;
        let now = Instant::now();
        let mut results = Vec::new();

        let timed_out: Vec<_> = pending
            .iter()
            .filter(|(_, p)| now.duration_since(p.sent_at) > self.config.ping_timeout)
            .map(|(seq, p)| (*seq, p.target.clone()))
            .collect();

        for (seq, target) in timed_out {
            pending.remove(&seq);
            results.push((target, true));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swim_config_default() {
        let config = SwimConfig::default();
        assert_eq!(config.ping_interval, Duration::from_millis(500));
        assert_eq!(config.ping_timeout, Duration::from_millis(200));
        assert_eq!(config.indirect_probes, 3);
    }

    #[tokio::test]
    async fn test_swim_detector_ping() {
        let node = NodeId::generate();
        let detector = SwimDetector::new(node.clone(), SwimConfig::default());

        let (seq, ping) = detector.create_ping();
        assert_eq!(seq, 0);

        match ping {
            SwimMessage::Ping { seq: s, from } => {
                assert_eq!(s, 0);
                assert_eq!(from, node);
            }
            _ => panic!("Expected Ping message"),
        }
    }

    #[tokio::test]
    async fn test_swim_detector_ack() {
        let node = NodeId::generate();
        let detector = SwimDetector::new(node.clone(), SwimConfig::default());

        let ack = detector.create_ack(42);
        match ack {
            SwimMessage::Ack { seq, from } => {
                assert_eq!(seq, 42);
                assert_eq!(from, node);
            }
            _ => panic!("Expected Ack message"),
        }
    }

    #[tokio::test]
    async fn test_swim_pending_ping() {
        let node = NodeId::generate();
        let target = NodeId::generate();
        let detector = SwimDetector::new(node, SwimConfig::default());

        detector.ping_sent(1, target).await;
        detector.ack_received(1).await;

        // Should have no pending pings
        let timeouts = detector.check_timeouts().await;
        assert!(timeouts.is_empty());
    }
}
