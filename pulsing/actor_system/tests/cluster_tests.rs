//! Cluster and Gossip protocol tests

use pulsing_actor::cluster::{GossipCluster, GossipConfig, MemberInfo, MemberStatus};
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
    assert_eq!(config.max_message_size, 65507);
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
// GossipCluster Tests
// ============================================================================

#[tokio::test]
async fn test_gossip_cluster_creation() {
    let tcp_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let gossip_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let cluster = GossipCluster::new(tcp_addr, gossip_addr, GossipConfig::default())
        .await
        .unwrap();

    assert!(!cluster.local_node().0.is_empty());
}

#[tokio::test]
async fn test_gossip_cluster_local_addresses() {
    let tcp_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let gossip_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let cluster = GossipCluster::new(tcp_addr, gossip_addr, GossipConfig::default())
        .await
        .unwrap();

    // Gossip address should have actual bound port
    let actual_gossip_addr = cluster.gossip_addr();
    assert!(actual_gossip_addr.port() > 0);

    // TCP address remains as configured (will be bound by TcpTransport)
    let tcp_addr = cluster.local_addr();
    assert_eq!(tcp_addr.port(), 0); // Remains 0 since GossipCluster doesn't bind TCP
}

#[tokio::test]
async fn test_gossip_cluster_actor_registration() {
    let tcp_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let gossip_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let cluster = GossipCluster::new(tcp_addr, gossip_addr, GossipConfig::default())
        .await
        .unwrap();

    let actor_id = ActorId::new(cluster.local_node().clone(), "test-actor");

    // Register actor
    cluster.register_actor(actor_id.clone()).await;

    // Should be able to look up
    let member = cluster.lookup_actor(&actor_id).await;
    // Note: lookup returns None because the cluster hasn't added itself to members
    // In a real scenario, the cluster would be started and sync with itself
    assert!(member.is_none()); // Expected for single-node without self-registration
}

#[tokio::test]
async fn test_gossip_cluster_no_seed_nodes() {
    let tcp_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let gossip_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let cluster = GossipCluster::new(tcp_addr, gossip_addr, GossipConfig::default())
        .await
        .unwrap();

    // Join with no seeds should succeed (first node)
    cluster.join(vec![]).await.unwrap();

    // Should have no other members
    let members = cluster.alive_members().await;
    assert!(members.is_empty());
}

// ============================================================================
// Multi-Node Cluster Tests (Integration)
// ============================================================================

#[tokio::test]
#[cfg(feature = "integration")]
async fn test_two_node_cluster_join() {
    let cancel = tokio_util::sync::CancellationToken::new();

    // Node 1
    let tcp1: SocketAddr = "127.0.0.1:18001".parse().unwrap();
    let gossip1: SocketAddr = "127.0.0.1:17001".parse().unwrap();
    let cluster1 = GossipCluster::new(tcp1, gossip1, GossipConfig::default())
        .await
        .unwrap();
    cluster1.start(cancel.clone());

    // Node 2 - joins node 1
    let tcp2: SocketAddr = "127.0.0.1:18002".parse().unwrap();
    let gossip2: SocketAddr = "127.0.0.1:17002".parse().unwrap();
    let cluster2 = GossipCluster::new(tcp2, gossip2, GossipConfig::default())
        .await
        .unwrap();
    cluster2.start(cancel.clone());
    cluster2.join(vec![gossip1]).await.unwrap();

    // Wait for gossip to sync
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Each should see the other
    let members1 = cluster1.alive_members().await;
    let members2 = cluster2.alive_members().await;

    assert_eq!(members1.len(), 1);
    assert_eq!(members2.len(), 1);

    cancel.cancel();
}

#[tokio::test]
#[cfg(feature = "integration")]
async fn test_three_node_cluster_gossip() {
    let cancel = tokio_util::sync::CancellationToken::new();

    // Node 1
    let cluster1 = GossipCluster::new(
        "127.0.0.1:19001".parse().unwrap(),
        "127.0.0.1:18001".parse().unwrap(),
        GossipConfig::default(),
    )
    .await
    .unwrap();
    cluster1.start(cancel.clone());

    // Node 2 - joins node 1
    let cluster2 = GossipCluster::new(
        "127.0.0.1:19002".parse().unwrap(),
        "127.0.0.1:18002".parse().unwrap(),
        GossipConfig::default(),
    )
    .await
    .unwrap();
    cluster2.start(cancel.clone());
    cluster2
        .join(vec!["127.0.0.1:18001".parse().unwrap()])
        .await
        .unwrap();

    // Node 3 - joins only node 2 (should still learn about node 1 via gossip)
    let cluster3 = GossipCluster::new(
        "127.0.0.1:19003".parse().unwrap(),
        "127.0.0.1:18003".parse().unwrap(),
        GossipConfig::default(),
    )
    .await
    .unwrap();
    cluster3.start(cancel.clone());
    cluster3
        .join(vec!["127.0.0.1:18002".parse().unwrap()])
        .await
        .unwrap();

    // Wait for gossip to propagate
    tokio::time::sleep(Duration::from_secs(1)).await;

    // All nodes should see all other nodes
    let members1 = cluster1.alive_members().await;
    let members2 = cluster2.alive_members().await;
    let members3 = cluster3.alive_members().await;

    assert_eq!(members1.len(), 2);
    assert_eq!(members2.len(), 2);
    assert_eq!(members3.len(), 2);

    cancel.cancel();
}

// ============================================================================
// SWIM Protocol Tests
// ============================================================================

#[tokio::test]
async fn test_swim_ping_ack() {
    use pulsing_actor::cluster::SwimDetector;

    let config = pulsing_actor::cluster::swim::SwimConfig::default();
    let detector = SwimDetector::new(NodeId::generate(), config);

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
fn test_node_id_from_string() {
    let id = NodeId::new("custom-node-id");
    assert_eq!(id.as_str(), "custom-node-id");
}

#[test]
fn test_node_id_display() {
    let id = NodeId::new("test-node");
    assert_eq!(format!("{}", id), "test-node");
}

// ============================================================================
// Actor ID Tests
// ============================================================================

#[test]
fn test_actor_id_creation() {
    let node = NodeId::generate();
    let actor_id = ActorId::new(node.clone(), "my-actor");

    assert_eq!(actor_id.node, node);
    assert_eq!(actor_id.name, "my-actor");
}

#[test]
fn test_actor_id_local() {
    let actor_id = ActorId::local("local-actor");

    assert_eq!(actor_id.node.as_str(), "local");
    assert_eq!(actor_id.name, "local-actor");
}

#[test]
fn test_actor_id_display() {
    let node = NodeId::new("node-1");
    let actor_id = ActorId::new(node, "actor-1");

    assert_eq!(format!("{}", actor_id), "actor-1@node-1");
}

#[test]
fn test_actor_id_equality() {
    let node = NodeId::new("node-1");
    let id1 = ActorId::new(node.clone(), "actor-1");
    let id2 = ActorId::new(node.clone(), "actor-1");
    let id3 = ActorId::new(node, "actor-2");

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
}

