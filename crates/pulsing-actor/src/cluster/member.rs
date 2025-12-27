//! Cluster member types

use crate::actor::{ActorId, ActorPath, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::SocketAddr;
use std::time::Instant;

// ============================================================================
// New Gossip Protocol (Redis Cluster style)
// ============================================================================

/// Node status in the new gossip protocol (Redis Cluster style)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is online and healthy
    Online = 0,
    /// Node is possibly failed (local detection, not confirmed)
    PFail = 1,
    /// Node is confirmed failed (majority of nodes agree)
    Fail = 2,
    /// Node is in handshake (new node joining)
    Handshake = 3,
}

impl NodeStatus {
    pub fn is_online(&self) -> bool {
        matches!(self, Self::Online)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, Self::PFail | Self::Fail)
    }
}

/// Cluster node information (new format)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterNode {
    /// Node identifier
    pub node_id: NodeId,
    /// Network address
    pub addr: SocketAddr,
    /// Current status
    pub status: NodeStatus,
    /// Configuration epoch (for conflict resolution)
    pub epoch: u64,
    /// Last seen timestamp (milliseconds since epoch)
    pub last_seen: u64,
}

impl ClusterNode {
    pub fn new(node_id: NodeId, addr: SocketAddr, epoch: u64) -> Self {
        Self {
            node_id,
            addr,
            status: NodeStatus::Online,
            epoch,
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Check if this node info supersedes another (based on epoch)
    pub fn supersedes(&self, other: &ClusterNode) -> bool {
        // Higher epoch always wins
        if self.epoch != other.epoch {
            return self.epoch > other.epoch;
        }
        // Same epoch: Fail > PFail > Online
        self.status > other.status
    }
}

/// Failure information to propagate via gossip
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailureInfo {
    /// Node that failed
    pub node_id: NodeId,
    /// Failure status (PFail or Fail)
    pub status: NodeStatus,
    /// Epoch when failure was detected
    pub epoch: u64,
    /// Node that reported the failure
    pub reported_by: NodeId,
}

// ============================================================================
// Legacy types (kept for backward compatibility)
// ============================================================================

/// Member status in the cluster
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberStatus {
    /// Member is alive and healthy
    Alive,
    /// Member is suspected to be down (not responding to pings)
    Suspect,
    /// Member is confirmed dead
    Dead,
    /// Member is leaving the cluster gracefully
    Leaving,
}

impl MemberStatus {
    pub fn is_alive(&self) -> bool {
        matches!(self, Self::Alive)
    }

    pub fn is_reachable(&self) -> bool {
        matches!(self, Self::Alive | Self::Suspect)
    }
}

/// Information about a cluster member
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemberInfo {
    /// Node identifier
    pub node_id: NodeId,

    /// Network address (for TCP communication)
    pub addr: SocketAddr,

    /// Gossip address (for UDP gossip)
    pub gossip_addr: SocketAddr,

    /// Current status
    pub status: MemberStatus,

    /// Incarnation number (for conflict resolution)
    /// Higher incarnation wins in case of conflicting information
    pub incarnation: u64,

    /// Timestamp of last update (not serialized, local only)
    #[serde(skip)]
    pub last_update: Option<Instant>,
}

impl MemberInfo {
    /// Create a new member info
    pub fn new(node_id: NodeId, addr: SocketAddr, gossip_addr: SocketAddr) -> Self {
        Self {
            node_id,
            addr,
            gossip_addr,
            status: MemberStatus::Alive,
            incarnation: 0,
            last_update: Some(Instant::now()),
        }
    }

    /// Update incarnation number (used when refuting suspicion)
    pub fn refute(&mut self) {
        self.incarnation += 1;
        self.status = MemberStatus::Alive;
        self.last_update = Some(Instant::now());
    }

    /// Mark as suspect
    pub fn suspect(&mut self) {
        if self.status == MemberStatus::Alive {
            self.status = MemberStatus::Suspect;
            self.last_update = Some(Instant::now());
        }
    }

    /// Mark as dead
    pub fn mark_dead(&mut self) {
        self.status = MemberStatus::Dead;
        self.last_update = Some(Instant::now());
    }

    /// Check if this info supersedes another (based on incarnation)
    pub fn supersedes(&self, other: &MemberInfo) -> bool {
        // Higher incarnation always wins
        if self.incarnation != other.incarnation {
            return self.incarnation > other.incarnation;
        }

        // Same incarnation: Dead > Suspect > Alive
        matches!(
            (&self.status, &other.status),
            (MemberStatus::Dead, _) | (MemberStatus::Suspect, MemberStatus::Alive)
        )
    }
}

impl PartialEq for MemberInfo {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for MemberInfo {}

impl std::hash::Hash for MemberInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node_id.hash(state);
    }
}

/// Actor location in the cluster (legacy, for backward compatibility)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ActorLocation {
    /// Actor identifier
    pub actor_id: ActorId,

    /// Node where the actor resides
    pub node_id: NodeId,

    /// Version for conflict resolution
    pub version: u64,
}

impl ActorLocation {
    pub fn new(actor_id: ActorId, node_id: NodeId) -> Self {
        Self {
            actor_id,
            node_id,
            version: 0,
        }
    }
}

/// Named actor registration info - supports multiple instances
///
/// A named actor can have multiple instances across different nodes.
/// The registry tracks all instances and supports load-balanced access.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedActorInfo {
    /// Actor path (namespace/path/name)
    pub path: ActorPath,

    /// All instances (node IDs where this actor is deployed)
    pub instances: HashSet<NodeId>,

    /// Version number for conflict resolution (CRDT-like merge)
    pub version: u64,
}

impl NamedActorInfo {
    /// Create a new named actor info
    pub fn new(path: ActorPath) -> Self {
        Self {
            path,
            instances: HashSet::new(),
            version: 0,
        }
    }

    /// Create with a single instance
    pub fn with_instance(path: ActorPath, node_id: NodeId) -> Self {
        let mut instances = HashSet::new();
        instances.insert(node_id);
        Self {
            path,
            instances,
            version: 1,
        }
    }

    /// Add an instance
    pub fn add_instance(&mut self, node_id: NodeId) {
        if self.instances.insert(node_id) {
            self.version += 1;
        }
    }

    /// Remove an instance
    pub fn remove_instance(&mut self, node_id: &NodeId) -> bool {
        if self.instances.remove(node_id) {
            self.version += 1;
            true
        } else {
            false
        }
    }

    /// Check if the actor has any instances
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Get the number of instances
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Merge with another NamedActorInfo (union of instances)
    pub fn merge(&mut self, other: &NamedActorInfo) {
        for node_id in &other.instances {
            self.instances.insert(*node_id);
        }
        self.version = self.version.max(other.version) + 1;
    }

    /// Select a random instance for load balancing
    pub fn select_instance(&self) -> Option<NodeId> {
        use rand::prelude::IteratorRandom;
        let mut rng = rand::rng();
        self.instances.iter().choose(&mut rng).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_member_supersedes() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let mut m1 = MemberInfo::new(node_id, addr, addr);
        let mut m2 = MemberInfo::new(node_id, addr, addr);

        // Same incarnation, same status - neither supersedes
        assert!(!m1.supersedes(&m2));
        assert!(!m2.supersedes(&m1));

        // Suspect supersedes Alive at same incarnation
        m1.suspect();
        assert!(m1.supersedes(&m2));
        assert!(!m2.supersedes(&m1));

        // Higher incarnation always wins
        m2.incarnation = 1;
        assert!(!m1.supersedes(&m2));
        assert!(m2.supersedes(&m1));
    }

    #[test]
    fn test_member_refute() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let mut member = MemberInfo::new(node_id, addr, addr);
        member.suspect();
        assert_eq!(member.status, MemberStatus::Suspect);

        member.refute();
        assert_eq!(member.status, MemberStatus::Alive);
        assert_eq!(member.incarnation, 1);
    }
}
