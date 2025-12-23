//! Cluster and Gossip protocol tests

use pulsing_actor::actor::{ActorId, NodeId};
use pulsing_actor::cluster::{GossipConfig, MemberInfo, MemberStatus};
use pulsing_actor::prelude::*;
use std::net::SocketAddr;
use std::time::Duration;

// ============================================================================
// Gossip Protocol Configuration Tests
// ============================================================================

#[test]
fn test_gossip_config_default() {
    let config = GossipConfig::default();

    assert_eq!(config.gossip_interval, Duration::from_millis(200));
    assert_eq!(config.fanout, 3);
}

// ============================================================================
// Member Status Tests
// ============================================================================

#[test]
fn test_member_status_alive() {
    let status = MemberStatus::Alive;
    assert!(status.is_alive());
    assert!(status.is_reachable());
}

#[test]
fn test_member_status_suspect() {
    let status = MemberStatus::Suspect;
    assert!(!status.is_alive());
    assert!(status.is_reachable()); // Suspect is still reachable
}

#[test]
fn test_member_status_dead() {
    let status = MemberStatus::Dead;
    assert!(!status.is_alive());
    assert!(!status.is_reachable());
}

#[test]
fn test_member_status_leaving() {
    let status = MemberStatus::Leaving;
    assert!(!status.is_alive());
    assert!(!status.is_reachable());
}

// ============================================================================
// MemberInfo Tests
// ============================================================================

#[test]
fn test_member_info_creation() {
    let node_id = NodeId::generate();
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();
    let gossip_addr: SocketAddr = "127.0.0.1:7000".parse().unwrap();

    let member = MemberInfo::new(node_id.clone(), addr, gossip_addr);

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
fn test_member_info_supersedes_by_incarnation() {
    let node_id = NodeId::generate();
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

    let mut m1 = MemberInfo::new(node_id.clone(), addr, addr);
    let m2 = MemberInfo::new(node_id, addr, addr);

    // Same incarnation - neither supersedes
    assert!(!m1.supersedes(&m2));
    assert!(!m2.supersedes(&m1));

    // Higher incarnation supersedes
    m1.incarnation = 1;
    assert!(m1.supersedes(&m2));
    assert!(!m2.supersedes(&m1));
}

#[test]
fn test_member_info_supersedes_by_status() {
    let node_id = NodeId::generate();
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

    let mut alive = MemberInfo::new(node_id.clone(), addr, addr);
    let mut suspect = MemberInfo::new(node_id.clone(), addr, addr);
    let mut dead = MemberInfo::new(node_id, addr, addr);

    alive.status = MemberStatus::Alive;
    suspect.status = MemberStatus::Suspect;
    dead.status = MemberStatus::Dead;

    // Dead supersedes all
    assert!(dead.supersedes(&alive));
    assert!(dead.supersedes(&suspect));

    // Suspect supersedes Alive
    assert!(suspect.supersedes(&alive));

    // Alive doesn't supersede others
    assert!(!alive.supersedes(&suspect));
    assert!(!alive.supersedes(&dead));
}

// ============================================================================
// ActorSystem Cluster Tests
// ============================================================================

#[tokio::test]
async fn test_system_with_cluster_config() {
    let config = SystemConfig::standalone();
    let system = pulsing_actor::system::ActorSystem::new(config)
        .await
        .unwrap();

    assert!(!system.node_id().0.is_empty());
    assert!(system.addr().port() > 0);

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_addr_is_consistent() {
    let config = SystemConfig::standalone();
    let system = pulsing_actor::system::ActorSystem::new(config)
        .await
        .unwrap();

    // tcp_addr and gossip_addr are now unified into a single addr()
    // Both actor messages and gossip use the same HTTP/2 transport
    assert_ne!(system.addr().port(), 0);

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_with_specific_addr() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let config = SystemConfig::with_addr(addr);
    let system = pulsing_actor::system::ActorSystem::new(config)
        .await
        .unwrap();

    // Should have bound to an actual port
    assert!(system.addr().port() > 0);

    system.shutdown().await.unwrap();
}

// ============================================================================
// Actor ID Tests
// ============================================================================

#[test]
fn test_actor_id_creation() {
    let node_id = NodeId::generate();
    let actor_id = ActorId::new(node_id.clone(), "my-actor");

    assert_eq!(actor_id.node, node_id);
    assert_eq!(actor_id.name, "my-actor");
}

#[test]
fn test_actor_id_local() {
    let actor_id = ActorId::local("test-actor");

    assert_eq!(actor_id.node.0, "local");
    assert_eq!(actor_id.name, "test-actor");
}

#[test]
fn test_actor_id_equality() {
    let node_id = NodeId::generate();
    let id1 = ActorId::new(node_id.clone(), "actor");
    let id2 = ActorId::new(node_id, "actor");

    assert_eq!(id1, id2);
}

#[test]
fn test_actor_id_display() {
    let node_id = NodeId::generate();
    let actor_id = ActorId::new(node_id.clone(), "my-actor");
    let display = format!("{}", actor_id);

    assert!(display.contains("my-actor"));
    assert!(display.contains(&node_id.0));
}

// ============================================================================
// Node ID Tests
// ============================================================================

#[test]
fn test_node_id_generation() {
    let id1 = NodeId::generate();
    let id2 = NodeId::generate();

    // Should be unique
    assert_ne!(id1, id2);
    assert!(!id1.0.is_empty());
    assert!(!id2.0.is_empty());
}

#[test]
fn test_node_id_display() {
    let node_id = NodeId::generate();
    let display = format!("{}", node_id);

    assert!(!display.is_empty());
    assert!(display.len() > 10); // UUID should be reasonably long
}

#[test]
fn test_node_id_from_string() {
    let original = NodeId::generate();
    let as_string = original.0.clone();
    let reconstructed = NodeId(as_string.clone());

    assert_eq!(original, reconstructed);
}

// ============================================================================
// SWIM Protocol Tests
// ============================================================================

#[tokio::test]
async fn test_swim_ping_ack() {
    use pulsing_actor::cluster::swim::{SwimConfig, SwimDetector};

    let node = NodeId::generate();
    let detector = SwimDetector::new(node.clone(), SwimConfig::default());

    let (seq, ping) = detector.create_ping();
    assert!(matches!(
        ping,
        pulsing_actor::cluster::swim::SwimMessage::Ping { .. }
    ));

    let target = NodeId::generate();
    detector.ping_sent(seq, target.clone()).await;

    // ack_received removes the pending ping
    detector.ack_received(seq).await;

    // After ack, check_timeouts should return empty (ping was acknowledged)
    let timeouts = detector.check_timeouts().await;
    assert!(timeouts.is_empty());
}

#[tokio::test]
async fn test_swim_timeout_detection() {
    use pulsing_actor::cluster::swim::{SwimConfig, SwimDetector};

    let config = SwimConfig {
        ping_timeout: Duration::from_millis(50),
        suspicion_timeout: Duration::from_millis(100),
        ..Default::default()
    };
    let detector = SwimDetector::new(NodeId::generate(), config);

    let (seq, _) = detector.create_ping();
    let target = NodeId::generate();
    detector.ping_sent(seq, target.clone()).await;

    // Wait for ping timeout
    tokio::time::sleep(Duration::from_millis(60)).await;

    let timeouts = detector.check_timeouts().await;
    assert!(!timeouts.is_empty());

    // Current simplified implementation directly suspects on timeout
    let (node, should_suspect) = &timeouts[0];
    assert_eq!(node, &target);
    assert!(should_suspect); // true = node should be suspected
}
