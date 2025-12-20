//! Cluster member types

use crate::actor::{ActorId, NodeId};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Instant;

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
        match (&self.status, &other.status) {
            (MemberStatus::Dead, _) => true,
            (MemberStatus::Suspect, MemberStatus::Alive) => true,
            _ => false,
        }
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

/// Actor location in the cluster
#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_member_supersedes() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let mut m1 = MemberInfo::new(node_id.clone(), addr, addr);
        let mut m2 = MemberInfo::new(node_id.clone(), addr, addr);

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

