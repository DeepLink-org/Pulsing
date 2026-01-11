//! Cluster and Gossip protocol tests
//!
//! This file contains integration tests for cluster functionality.
//! Unit tests for member types are in tests/cluster/member_tests.rs

mod cluster;

use pulsing_actor::actor::{ActorId, NodeId};
use pulsing_actor::cluster::GossipConfig;
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
// ActorSystem Cluster Tests
// ============================================================================

#[tokio::test]
async fn test_system_with_cluster_config() {
    let config = SystemConfig::standalone();
    let system = pulsing_actor::system::ActorSystem::new(config)
        .await
        .unwrap();

    assert!(system.node_id().0 != 0); // 0 is reserved for LOCAL
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
    let actor_id = ActorId::new(node_id, 123);

    assert_eq!(actor_id.node(), node_id);
    assert_eq!(actor_id.local_id(), 123);
}

#[test]
fn test_actor_id_local() {
    let actor_id = ActorId::local(456);

    assert!(actor_id.node().is_local());
    assert_eq!(actor_id.local_id(), 456);
}

#[test]
fn test_actor_id_equality() {
    let node_id = NodeId::generate();
    let id1 = ActorId::new(node_id, 1);
    let id2 = ActorId::new(node_id, 1);

    assert_eq!(id1, id2);
}

#[test]
fn test_actor_id_display() {
    let node_id = NodeId::generate();
    let actor_id = ActorId::new(node_id, 42);
    let display = format!("{}", actor_id);

    // Display format is "node_id:local_id"
    assert!(display.contains("42"));
    assert!(display.contains(&node_id.0.to_string()));
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
    assert!(id1.0 != 0); // 0 is reserved for LOCAL
    assert!(id2.0 != 0);
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
    let as_string = original.0;
    let reconstructed = NodeId(as_string);

    assert_eq!(original, reconstructed);
}

// ============================================================================
// SWIM Protocol Tests
// ============================================================================

#[tokio::test]
async fn test_swim_ping_ack() {
    use pulsing_actor::cluster::swim::{SwimConfig, SwimDetector};

    let node = NodeId::generate();
    let detector = SwimDetector::new(node, SwimConfig::default());

    let (seq, ping) = detector.create_ping();
    assert!(matches!(
        ping,
        pulsing_actor::cluster::swim::SwimMessage::Ping { .. }
    ));

    let target = NodeId::generate();
    detector.ping_sent(seq, target).await;

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
    detector.ping_sent(seq, target).await;

    // Wait for ping timeout
    tokio::time::sleep(Duration::from_millis(60)).await;

    let timeouts = detector.check_timeouts().await;
    assert!(!timeouts.is_empty());

    // Current simplified implementation directly suspects on timeout
    let (node, should_suspect) = &timeouts[0];
    assert_eq!(node, &target);
    assert!(should_suspect); // true = node should be suspected
}
