//! Cluster member types

use crate::actor::{ActorId, ActorPath, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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

/// Instance details for a named actor on a specific node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedActorInstance {
    /// The node ID where this instance is running
    pub node_id: NodeId,
    /// The actor ID of this instance
    pub actor_id: ActorId,
    /// Metadata (e.g., Python class, module, file path)
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl NamedActorInstance {
    /// Create a new instance with just node_id and actor_id
    pub fn new(node_id: NodeId, actor_id: ActorId) -> Self {
        Self {
            node_id,
            actor_id,
            metadata: HashMap::new(),
        }
    }

    /// Create with metadata
    pub fn with_metadata(
        node_id: NodeId,
        actor_id: ActorId,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            node_id,
            actor_id,
            metadata,
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

    /// All instances mapped by node_id
    pub instances: HashMap<NodeId, NamedActorInstance>,

    /// Legacy: just node IDs for backward compatibility
    #[serde(default)]
    pub instance_nodes: HashSet<NodeId>,

    /// Version number for conflict resolution (CRDT-like merge)
    pub version: u64,
}

impl NamedActorInfo {
    /// Create a new named actor info
    pub fn new(path: ActorPath) -> Self {
        Self {
            path,
            instances: HashMap::new(),
            instance_nodes: HashSet::new(),
            version: 0,
        }
    }

    /// Create with a single instance (legacy, no actor_id)
    pub fn with_instance(path: ActorPath, node_id: NodeId) -> Self {
        let mut instance_nodes = HashSet::new();
        instance_nodes.insert(node_id);
        Self {
            path,
            instances: HashMap::new(),
            instance_nodes,
            version: 1,
        }
    }

    /// Create with full instance details
    pub fn with_full_instance(path: ActorPath, instance: NamedActorInstance) -> Self {
        let mut instances = HashMap::new();
        let mut instance_nodes = HashSet::new();
        let node_id = instance.node_id;
        instance_nodes.insert(node_id);
        instances.insert(node_id, instance);
        Self {
            path,
            instances,
            instance_nodes,
            version: 1,
        }
    }

    /// Add an instance (legacy, no actor_id)
    pub fn add_instance(&mut self, node_id: NodeId) {
        if self.instance_nodes.insert(node_id) {
            self.version += 1;
        }
    }

    /// Add a full instance with details
    pub fn add_full_instance(&mut self, instance: NamedActorInstance) {
        let node_id = instance.node_id;
        self.instance_nodes.insert(node_id);
        self.instances.insert(node_id, instance);
        self.version += 1;
    }

    /// Remove an instance
    pub fn remove_instance(&mut self, node_id: &NodeId) -> bool {
        let removed_node = self.instance_nodes.remove(node_id);
        let removed_instance = self.instances.remove(node_id).is_some();
        if removed_node || removed_instance {
            self.version += 1;
            true
        } else {
            false
        }
    }

    /// Check if the actor has any instances
    pub fn is_empty(&self) -> bool {
        self.instance_nodes.is_empty() && self.instances.is_empty()
    }

    /// Get the number of instances
    pub fn instance_count(&self) -> usize {
        // Use instance_nodes for count (backward compatible)
        self.instance_nodes.len()
    }

    /// Get all node IDs where this actor has instances
    pub fn node_ids(&self) -> impl Iterator<Item = &NodeId> {
        self.instance_nodes.iter()
    }

    /// Get instance details for a node
    pub fn get_instance(&self, node_id: &NodeId) -> Option<&NamedActorInstance> {
        self.instances.get(node_id)
    }

    /// Merge with another NamedActorInfo (union of instances)
    pub fn merge(&mut self, other: &NamedActorInfo) {
        for node_id in &other.instance_nodes {
            self.instance_nodes.insert(*node_id);
        }
        for (node_id, instance) in &other.instances {
            self.instances.insert(*node_id, instance.clone());
        }
        self.version = self.version.max(other.version) + 1;
    }

    /// Select a random instance for load balancing
    pub fn select_instance(&self) -> Option<NodeId> {
        use rand::prelude::IteratorRandom;
        let mut rng = rand::rng();
        self.instance_nodes.iter().choose(&mut rng).cloned()
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

    #[test]
    fn test_node_status() {
        assert!(NodeStatus::Online.is_online());
        assert!(!NodeStatus::PFail.is_online());
        assert!(!NodeStatus::Fail.is_online());
        assert!(!NodeStatus::Handshake.is_online());

        assert!(!NodeStatus::Online.is_failed());
        assert!(NodeStatus::PFail.is_failed());
        assert!(NodeStatus::Fail.is_failed());
        assert!(!NodeStatus::Handshake.is_failed());
    }

    #[test]
    fn test_member_status() {
        assert!(MemberStatus::Alive.is_alive());
        assert!(!MemberStatus::Suspect.is_alive());
        assert!(!MemberStatus::Dead.is_alive());
        assert!(!MemberStatus::Leaving.is_alive());

        assert!(MemberStatus::Alive.is_reachable());
        assert!(MemberStatus::Suspect.is_reachable());
        assert!(!MemberStatus::Dead.is_reachable());
        assert!(!MemberStatus::Leaving.is_reachable());
    }

    #[test]
    fn test_cluster_node_new() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let node = ClusterNode::new(node_id, addr, 1);
        assert_eq!(node.node_id, node_id);
        assert_eq!(node.addr, addr);
        assert_eq!(node.status, NodeStatus::Online);
        assert_eq!(node.epoch, 1);
        assert!(node.last_seen > 0);
    }

    #[test]
    fn test_cluster_node_supersedes() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let mut n1 = ClusterNode::new(node_id, addr, 1);
        let mut n2 = ClusterNode::new(node_id, addr, 1);

        // Same epoch, same status - neither supersedes
        assert!(!n1.supersedes(&n2));
        assert!(!n2.supersedes(&n1));

        // Higher epoch wins
        n2.epoch = 2;
        assert!(n2.supersedes(&n1));
        assert!(!n1.supersedes(&n2));

        // Same epoch, higher status wins
        n1.epoch = 2;
        n1.status = NodeStatus::Fail;
        assert!(n1.supersedes(&n2));
    }

    #[test]
    fn test_member_info_mark_dead() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let mut member = MemberInfo::new(node_id, addr, addr);
        assert_eq!(member.status, MemberStatus::Alive);

        member.mark_dead();
        assert_eq!(member.status, MemberStatus::Dead);
    }

    #[test]
    fn test_member_info_suspect_from_alive() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let mut member = MemberInfo::new(node_id, addr, addr);
        assert_eq!(member.status, MemberStatus::Alive);

        member.suspect();
        assert_eq!(member.status, MemberStatus::Suspect);
    }

    #[test]
    fn test_member_info_suspect_already_suspect() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let mut member = MemberInfo::new(node_id, addr, addr);
        member.suspect();
        member.suspect(); // Should not change
        assert_eq!(member.status, MemberStatus::Suspect);
    }

    #[test]
    fn test_member_info_equality() {
        let node_id = NodeId::generate();
        let addr1: SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        let m1 = MemberInfo::new(node_id, addr1, addr1);
        let m2 = MemberInfo::new(node_id, addr2, addr2);

        // Equality is based on node_id only
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_member_info_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let node_id = NodeId::generate();
        let addr1: SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        let m1 = MemberInfo::new(node_id, addr1, addr1);
        let m2 = MemberInfo::new(node_id, addr2, addr2);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        m1.hash(&mut hasher1);
        m2.hash(&mut hasher2);

        // Same node_id should have same hash
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_actor_location() {
        let actor_id = ActorId::local(1);
        let node_id = NodeId::generate();

        let location = ActorLocation::new(actor_id, node_id);
        assert_eq!(location.actor_id, actor_id);
        assert_eq!(location.node_id, node_id);
        assert_eq!(location.version, 0);
    }

    #[test]
    fn test_named_actor_info_new() {
        let path = ActorPath::new("services/llm").unwrap();
        let info = NamedActorInfo::new(path.clone());

        assert_eq!(info.path, path);
        assert!(info.instances.is_empty());
        assert!(info.is_empty());
        assert_eq!(info.version, 0);
    }

    #[test]
    fn test_named_actor_info_with_instance() {
        let path = ActorPath::new("services/llm").unwrap();
        let node_id = NodeId::generate();

        let info = NamedActorInfo::with_instance(path.clone(), node_id);

        assert_eq!(info.path, path);
        assert_eq!(info.instance_count(), 1);
        assert!(!info.is_empty());
        assert_eq!(info.version, 1);
        assert!(info.instance_nodes.contains(&node_id));
    }

    #[test]
    fn test_named_actor_info_add_instance() {
        let path = ActorPath::new("services/llm").unwrap();
        let node1 = NodeId::generate();
        let node2 = NodeId::generate();

        let mut info = NamedActorInfo::new(path);
        info.add_instance(node1);
        info.add_instance(node2);

        assert_eq!(info.instance_count(), 2);
        assert_eq!(info.version, 2);
    }

    #[test]
    fn test_named_actor_info_add_duplicate_instance() {
        let path = ActorPath::new("services/llm").unwrap();
        let node_id = NodeId::generate();

        let mut info = NamedActorInfo::new(path);
        info.add_instance(node_id);
        info.add_instance(node_id); // Duplicate

        assert_eq!(info.instance_count(), 1);
        assert_eq!(info.version, 1); // Version not incremented for duplicate
    }

    #[test]
    fn test_named_actor_info_remove_instance() {
        let path = ActorPath::new("services/llm").unwrap();
        let node1 = NodeId::generate();
        let node2 = NodeId::generate();

        let mut info = NamedActorInfo::new(path);
        info.add_instance(node1);
        info.add_instance(node2);

        assert!(info.remove_instance(&node1));
        assert_eq!(info.instance_count(), 1);

        assert!(!info.remove_instance(&node1)); // Already removed
    }

    #[test]
    fn test_named_actor_info_merge() {
        let path = ActorPath::new("services/llm").unwrap();
        let node1 = NodeId::generate();
        let node2 = NodeId::generate();
        let node3 = NodeId::generate();

        let mut info1 = NamedActorInfo::with_instance(path.clone(), node1);
        info1.add_instance(node2);

        let mut info2 = NamedActorInfo::with_instance(path.clone(), node2);
        info2.add_instance(node3);

        info1.merge(&info2);

        assert_eq!(info1.instance_count(), 3);
        assert!(info1.instance_nodes.contains(&node1));
        assert!(info1.instance_nodes.contains(&node2));
        assert!(info1.instance_nodes.contains(&node3));
    }

    #[test]
    fn test_named_actor_info_select_instance() {
        let path = ActorPath::new("services/llm").unwrap();
        let node_id = NodeId::generate();

        let info = NamedActorInfo::with_instance(path, node_id);

        // Should return the only instance
        let selected = info.select_instance();
        assert_eq!(selected, Some(node_id));
    }

    #[test]
    fn test_named_actor_info_select_instance_empty() {
        let path = ActorPath::new("services/llm").unwrap();
        let info = NamedActorInfo::new(path);

        assert!(info.select_instance().is_none());
    }

    #[test]
    fn test_failure_info() {
        let node_id = NodeId::generate();
        let reporter_id = NodeId::generate();

        let failure = FailureInfo {
            node_id,
            status: NodeStatus::PFail,
            epoch: 5,
            reported_by: reporter_id,
        };

        assert_eq!(failure.node_id, node_id);
        assert_eq!(failure.status, NodeStatus::PFail);
        assert_eq!(failure.epoch, 5);
        assert_eq!(failure.reported_by, reporter_id);
    }

    #[test]
    fn test_member_supersedes_dead() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let alive = MemberInfo::new(node_id, addr, addr);
        let mut dead = MemberInfo::new(node_id, addr, addr);
        dead.mark_dead();

        // Dead supersedes Alive at same incarnation
        assert!(dead.supersedes(&alive));
        assert!(!alive.supersedes(&dead));
    }

    // ========================================================================
    // NamedActorInstance Tests
    // ========================================================================

    #[test]
    fn test_named_actor_instance_new() {
        let node_id = NodeId::generate();
        let actor_id = ActorId::local(42);

        let instance = NamedActorInstance::new(node_id, actor_id);

        assert_eq!(instance.node_id, node_id);
        assert_eq!(instance.actor_id, actor_id);
        assert!(instance.metadata.is_empty());
    }

    #[test]
    fn test_named_actor_instance_with_metadata() {
        let node_id = NodeId::generate();
        let actor_id = ActorId::local(42);
        let mut metadata = HashMap::new();
        metadata.insert("class".to_string(), "Counter".to_string());
        metadata.insert("module".to_string(), "__main__".to_string());
        metadata.insert("file".to_string(), "/app/main.py".to_string());

        let instance = NamedActorInstance::with_metadata(node_id, actor_id, metadata.clone());

        assert_eq!(instance.node_id, node_id);
        assert_eq!(instance.actor_id, actor_id);
        assert_eq!(instance.metadata.get("class"), Some(&"Counter".to_string()));
        assert_eq!(
            instance.metadata.get("module"),
            Some(&"__main__".to_string())
        );
        assert_eq!(
            instance.metadata.get("file"),
            Some(&"/app/main.py".to_string())
        );
    }

    #[test]
    fn test_named_actor_info_with_full_instance() {
        let path = ActorPath::new("actors/counter").unwrap();
        let node_id = NodeId::generate();
        let actor_id = ActorId::local(42);
        let mut metadata = HashMap::new();
        metadata.insert("class".to_string(), "Counter".to_string());

        let instance = NamedActorInstance::with_metadata(node_id, actor_id, metadata);
        let info = NamedActorInfo::with_full_instance(path.clone(), instance);

        assert_eq!(info.path, path);
        assert_eq!(info.instance_count(), 1);
        assert!(info.instance_nodes.contains(&node_id));
        assert!(info.instances.contains_key(&node_id));

        let retrieved = info.get_instance(&node_id).unwrap();
        assert_eq!(retrieved.actor_id, actor_id);
        assert_eq!(
            retrieved.metadata.get("class"),
            Some(&"Counter".to_string())
        );
    }

    #[test]
    fn test_named_actor_info_add_full_instance() {
        let path = ActorPath::new("actors/counter").unwrap();
        let node1 = NodeId::generate();
        let node2 = NodeId::generate();
        let actor_id1 = ActorId::local(1);
        let actor_id2 = ActorId::local(2);

        let mut info = NamedActorInfo::new(path);

        let instance1 = NamedActorInstance::new(node1, actor_id1);
        info.add_full_instance(instance1);
        assert_eq!(info.instance_count(), 1);

        let instance2 = NamedActorInstance::new(node2, actor_id2);
        info.add_full_instance(instance2);
        assert_eq!(info.instance_count(), 2);

        assert!(info.get_instance(&node1).is_some());
        assert!(info.get_instance(&node2).is_some());
        assert_eq!(info.get_instance(&node1).unwrap().actor_id, actor_id1);
        assert_eq!(info.get_instance(&node2).unwrap().actor_id, actor_id2);
    }

    #[test]
    fn test_named_actor_info_get_instance_not_found() {
        let path = ActorPath::new("actors/counter").unwrap();
        let node_id = NodeId::generate();

        let info = NamedActorInfo::new(path);

        assert!(info.get_instance(&node_id).is_none());
    }

    #[test]
    fn test_named_actor_info_merge_with_full_instances() {
        let path = ActorPath::new("actors/counter").unwrap();
        let node1 = NodeId::generate();
        let node2 = NodeId::generate();
        let actor_id1 = ActorId::local(1);
        let actor_id2 = ActorId::local(2);

        let mut metadata1 = HashMap::new();
        metadata1.insert("class".to_string(), "Counter".to_string());
        let instance1 = NamedActorInstance::with_metadata(node1, actor_id1, metadata1);
        let mut info1 = NamedActorInfo::with_full_instance(path.clone(), instance1);

        let mut metadata2 = HashMap::new();
        metadata2.insert("class".to_string(), "Counter".to_string());
        let instance2 = NamedActorInstance::with_metadata(node2, actor_id2, metadata2);
        let info2 = NamedActorInfo::with_full_instance(path.clone(), instance2);

        info1.merge(&info2);

        assert_eq!(info1.instance_count(), 2);
        assert!(info1.get_instance(&node1).is_some());
        assert!(info1.get_instance(&node2).is_some());
    }

    #[test]
    fn test_named_actor_info_node_ids_iterator() {
        let path = ActorPath::new("actors/counter").unwrap();
        let node1 = NodeId::generate();
        let node2 = NodeId::generate();

        let mut info = NamedActorInfo::new(path);
        info.add_instance(node1);
        info.add_instance(node2);

        let node_ids: Vec<_> = info.node_ids().collect();
        assert_eq!(node_ids.len(), 2);
        assert!(node_ids.contains(&&node1));
        assert!(node_ids.contains(&&node2));
    }
}
