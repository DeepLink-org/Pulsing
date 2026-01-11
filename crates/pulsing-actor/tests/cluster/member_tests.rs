//! Member types tests
//!
//! Tests for cluster member types: NodeStatus, MemberStatus, MemberInfo,
//! ClusterNode, NamedActorInfo, NamedActorInstance, etc.

use pulsing_actor::actor::{ActorId, ActorPath, NodeId};
use pulsing_actor::cluster::{
    ActorLocation, ClusterNode, FailureInfo, MemberInfo, MemberStatus, NamedActorInfo,
    NamedActorInstance, NodeStatus,
};
use std::collections::HashMap;
use std::net::SocketAddr;

// ============================================================================
// NodeStatus Tests
// ============================================================================

#[test]
fn test_node_status_is_online() {
    assert!(NodeStatus::Online.is_online());
    assert!(!NodeStatus::PFail.is_online());
    assert!(!NodeStatus::Fail.is_online());
    assert!(!NodeStatus::Handshake.is_online());
}

#[test]
fn test_node_status_is_failed() {
    assert!(!NodeStatus::Online.is_failed());
    assert!(NodeStatus::PFail.is_failed());
    assert!(NodeStatus::Fail.is_failed());
    assert!(!NodeStatus::Handshake.is_failed());
}

// ============================================================================
// MemberStatus Tests
// ============================================================================

#[test]
fn test_member_status_is_alive() {
    assert!(MemberStatus::Alive.is_alive());
    assert!(!MemberStatus::Suspect.is_alive());
    assert!(!MemberStatus::Dead.is_alive());
    assert!(!MemberStatus::Leaving.is_alive());
}

#[test]
fn test_member_status_is_reachable() {
    assert!(MemberStatus::Alive.is_reachable());
    assert!(MemberStatus::Suspect.is_reachable());
    assert!(!MemberStatus::Dead.is_reachable());
    assert!(!MemberStatus::Leaving.is_reachable());
}

// ============================================================================
// ClusterNode Tests
// ============================================================================

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
fn test_cluster_node_supersedes_by_epoch() {
    let node_id = NodeId::generate();
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

    let n1 = ClusterNode::new(node_id, addr, 1);
    let mut n2 = ClusterNode::new(node_id, addr, 1);

    // Same epoch, same status - neither supersedes
    assert!(!n1.supersedes(&n2));
    assert!(!n2.supersedes(&n1));

    // Higher epoch wins
    n2.epoch = 2;
    assert!(n2.supersedes(&n1));
    assert!(!n1.supersedes(&n2));
}

#[test]
fn test_cluster_node_supersedes_by_status() {
    let node_id = NodeId::generate();
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

    let mut n1 = ClusterNode::new(node_id, addr, 2);
    let n2 = ClusterNode::new(node_id, addr, 2);

    // Same epoch, higher status wins
    n1.status = NodeStatus::Fail;
    assert!(n1.supersedes(&n2));
}

// ============================================================================
// MemberInfo Tests
// ============================================================================

#[test]
fn test_member_info_creation() {
    let node_id = NodeId::generate();
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();
    let gossip_addr: SocketAddr = "127.0.0.1:7000".parse().unwrap();

    let member = MemberInfo::new(node_id, addr, gossip_addr);

    assert_eq!(member.node_id, node_id);
    assert_eq!(member.addr, addr);
    assert_eq!(member.gossip_addr, gossip_addr);
    assert_eq!(member.status, MemberStatus::Alive);
    assert_eq!(member.incarnation, 0);
}

#[test]
fn test_member_info_refute() {
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
fn test_member_info_mark_dead() {
    let node_id = NodeId::generate();
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

    let mut member = MemberInfo::new(node_id, addr, addr);
    assert_eq!(member.status, MemberStatus::Alive);

    member.mark_dead();
    assert_eq!(member.status, MemberStatus::Dead);
}

#[test]
fn test_member_info_supersedes_by_incarnation() {
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
fn test_member_info_supersedes_dead() {
    let node_id = NodeId::generate();
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

    let alive = MemberInfo::new(node_id, addr, addr);
    let mut dead = MemberInfo::new(node_id, addr, addr);
    dead.mark_dead();

    // Dead supersedes Alive at same incarnation
    assert!(dead.supersedes(&alive));
    assert!(!alive.supersedes(&dead));
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

// ============================================================================
// ActorLocation Tests
// ============================================================================

#[test]
fn test_actor_location() {
    let actor_id = ActorId::local(1);
    let node_id = NodeId::generate();

    let location = ActorLocation::new(actor_id, node_id);
    assert_eq!(location.actor_id, actor_id);
    assert_eq!(location.node_id, node_id);
    assert_eq!(location.version, 0);
}

// ============================================================================
// FailureInfo Tests
// ============================================================================

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

// ============================================================================
// NamedActorInstance Tests
// ============================================================================

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

// ============================================================================
// NamedActorInfo Tests
// ============================================================================

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
fn test_named_actor_info_get_instance_not_found() {
    let path = ActorPath::new("actors/counter").unwrap();
    let node_id = NodeId::generate();

    let info = NamedActorInfo::new(path);

    assert!(info.get_instance(&node_id).is_none());
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
