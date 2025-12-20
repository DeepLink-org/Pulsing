//! Multi-node cluster integration tests
//!
//! These tests verify cluster formation and basic cluster operations.

use pulsing_actor::prelude::*;
use pulsing_actor::system::SystemConfig;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Test Messages
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Ping {
    value: i32,
}

impl Message for Ping {
    fn type_id() -> &'static str {
        "Ping"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Pong {
    result: i32,
}

impl Message for Pong {
    fn type_id() -> &'static str {
        "Pong"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Increment;

impl Message for Increment {
    fn type_id() -> &'static str {
        "Increment"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetCount;

impl Message for GetCount {
    fn type_id() -> &'static str {
        "GetCount"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CountResponse {
    count: i32,
}

impl Message for CountResponse {
    fn type_id() -> &'static str {
        "CountResponse"
    }
}

// ============================================================================
// Test Actors
// ============================================================================

struct EchoActor {
    id: ActorId,
}

impl EchoActor {
    fn new(name: &str) -> Self {
        Self {
            id: ActorId::local(name),
        }
    }
}

#[async_trait]
impl Actor for EchoActor {
    fn id(&self) -> &ActorId {
        &self.id
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "Ping" => {
                let ping: Ping = msg.into_message()?;
                RawMessage::from_message(&Pong {
                    result: ping.value * 2,
                })
            }
            _ => Err(anyhow::anyhow!("Unknown message")),
        }
    }
}

struct CounterActor {
    id: ActorId,
    count: Arc<AtomicI32>,
}

impl CounterActor {
    fn new(name: &str, count: Arc<AtomicI32>) -> Self {
        Self {
            id: ActorId::local(name),
            count,
        }
    }
}

#[async_trait]
impl Actor for CounterActor {
    fn id(&self) -> &ActorId {
        &self.id
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "Increment" => {
                let new_count = self.count.fetch_add(1, Ordering::SeqCst) + 1;
                RawMessage::from_message(&CountResponse { count: new_count })
            }
            "GetCount" => {
                let count = self.count.load(Ordering::SeqCst);
                RawMessage::from_message(&CountResponse { count })
            }
            _ => Err(anyhow::anyhow!("Unknown message")),
        }
    }
}

// ============================================================================
// Cluster Setup Helpers
// ============================================================================

fn create_cluster_config(_port: u16) -> SystemConfig {
    // Use port 0 to let the OS assign an available port
    // This avoids port conflicts when tests run in parallel
    SystemConfig::with_addr("127.0.0.1:0".parse().unwrap())
}

// ============================================================================
// Two-Node Cluster Tests
// ============================================================================

mod two_node_tests {
    use super::*;

    #[tokio::test]
    async fn test_two_node_cluster_formation() {
        // Node 1 (seed)
        let config1 = create_cluster_config(20001);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        // Node 2 joins node 1
        let mut config2 = create_cluster_config(20002);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Wait for cluster formation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Both should be up
        assert!(!system1.cancel_token().is_cancelled());
        assert!(!system2.cancel_token().is_cancelled());

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actors_on_different_nodes() {
        // Node 1
        let config1 = create_cluster_config(20011);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        // Spawn actor on node 1
        let actor1 = EchoActor::new("echo");
        let actor1_ref = system1.spawn(actor1).await.unwrap();

        // Verify local actor works
        let response: Pong = actor1_ref.ask(Ping { value: 21 }).await.unwrap();
        assert_eq!(response.result, 42);

        // Node 2 joins
        let mut config2 = create_cluster_config(20012);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Spawn another actor on node 2
        let actor2 = EchoActor::new("echo2");
        let actor2_ref = system2.spawn(actor2).await.unwrap();

        // Wait for cluster sync
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Both local actors should work
        let response1: Pong = actor1_ref.ask(Ping { value: 10 }).await.unwrap();
        assert_eq!(response1.result, 20);

        let response2: Pong = actor2_ref.ask(Ping { value: 15 }).await.unwrap();
        assert_eq!(response2.result, 30);

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }
}

// ============================================================================
// Multi-Node Cluster Tests
// ============================================================================

mod multi_node_tests {
    use super::*;

    #[tokio::test]
    async fn test_three_node_cluster() {
        // Node 1 (seed)
        let config1 = create_cluster_config(20031);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        // Node 2
        let mut config2 = create_cluster_config(20032);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Node 3
        let mut config3 = create_cluster_config(20033);
        config3.seed_nodes = vec![gossip1_addr];
        let system3 = ActorSystem::new(config3).await.unwrap();

        // Wait for cluster formation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // All should be running
        assert!(!system1.cancel_token().is_cancelled());
        assert!(!system2.cancel_token().is_cancelled());
        assert!(!system3.cancel_token().is_cancelled());

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
        system3.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actors_on_multiple_nodes() {
        // Node 1
        let config1 = create_cluster_config(20041);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        let actor1 = EchoActor::new("actor-on-node1");
        let _ref1 = system1.spawn(actor1).await.unwrap();

        // Node 2
        let mut config2 = create_cluster_config(20042);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        let actor2 = EchoActor::new("actor-on-node2");
        let _ref2 = system2.spawn(actor2).await.unwrap();

        // Node 3
        let mut config3 = create_cluster_config(20043);
        config3.seed_nodes = vec![gossip1_addr];
        let system3 = ActorSystem::new(config3).await.unwrap();

        let actor3 = EchoActor::new("actor-on-node3");
        let _ref3 = system3.spawn(actor3).await.unwrap();

        // Each node has exactly one actor
        assert_eq!(system1.local_actor_names().len(), 1);
        assert_eq!(system2.local_actor_names().len(), 1);
        assert_eq!(system3.local_actor_names().len(), 1);

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
        system3.shutdown().await.unwrap();
    }
}

// ============================================================================
// Shared State Tests (via Actor)
// ============================================================================

mod shared_state_tests {
    use super::*;

    #[tokio::test]
    async fn test_shared_counter_single_node() {
        let count = Arc::new(AtomicI32::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter", count.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        // Multiple increments
        for _ in 0..100 {
            let _: CountResponse = actor_ref.ask(Increment).await.unwrap();
        }

        let response: CountResponse = actor_ref.ask(GetCount).await.unwrap();
        assert_eq!(response.count, 100);
        assert_eq!(count.load(Ordering::SeqCst), 100);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_increments() {
        let count = Arc::new(AtomicI32::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter", count.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        // Concurrent increments
        let mut handles = Vec::new();
        for _ in 0..50 {
            let ref_clone = actor_ref.clone();
            let handle = tokio::spawn(async move {
                for _ in 0..10 {
                    let _: CountResponse = ref_clone.ask(Increment).await.unwrap();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let response: CountResponse = actor_ref.ask(GetCount).await.unwrap();
        assert_eq!(response.count, 500); // 50 tasks * 10 increments

        system.shutdown().await.unwrap();
    }
}

// ============================================================================
// Node Failure Tests
// ============================================================================

mod failure_tests {
    use super::*;

    #[tokio::test]
    async fn test_graceful_shutdown() {
        let config1 = create_cluster_config(20051);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        let mut config2 = create_cluster_config(20052);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Wait for cluster
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Gracefully shutdown node 2
        system2.shutdown().await.unwrap();
        assert!(system2.cancel_token().is_cancelled());

        // Node 1 should still be running
        assert!(!system1.cancel_token().is_cancelled());

        system1.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_node_rejoin() {
        let config1 = create_cluster_config(20061);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        // Node 2 joins
        let mut config2 = create_cluster_config(20062);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Shutdown node 2
        system2.shutdown().await.unwrap();

        // Wait a bit for resources to be released
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Node 2 rejoins with a different port (simulating a new instance)
        // Using different port to avoid TIME_WAIT issues
        let mut config2_new = create_cluster_config(20063);
        config2_new.seed_nodes = vec![gossip1_addr];
        let system2_new = ActorSystem::new(config2_new).await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Both should be running
        assert!(!system1.cancel_token().is_cancelled());
        assert!(!system2_new.cancel_token().is_cancelled());

        system1.shutdown().await.unwrap();
        system2_new.shutdown().await.unwrap();
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_message_latency() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = EchoActor::new("echo");
        let actor_ref = system.spawn(actor).await.unwrap();

        // Warmup
        for _ in 0..100 {
            let _: Pong = actor_ref.ask(Ping { value: 1 }).await.unwrap();
        }

        // Measure latency
        let iterations: u32 = 1000;
        let start = std::time::Instant::now();

        for _ in 0..iterations {
            let _: Pong = actor_ref.ask(Ping { value: 1 }).await.unwrap();
        }

        let total = start.elapsed();
        let avg_latency = total / iterations;

        println!(
            "Average message latency: {:?} ({} iterations)",
            avg_latency, iterations
        );

        // Latency should be reasonable (< 1ms for local)
        assert!(avg_latency < Duration::from_millis(1));

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_throughput_benchmark() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = EchoActor::new("echo");
        let actor_ref = system.spawn(actor).await.unwrap();

        // Warmup
        for _ in 0..100 {
            let _: Pong = actor_ref.ask(Ping { value: 1 }).await.unwrap();
        }

        let duration = Duration::from_secs(1);
        let start = std::time::Instant::now();
        let mut count = 0u64;

        while start.elapsed() < duration {
            let _: Pong = actor_ref.ask(Ping { value: 1 }).await.unwrap();
            count += 1;
        }

        let actual_time = start.elapsed();
        let throughput = count as f64 / actual_time.as_secs_f64();

        println!(
            "Throughput: {:.2} msg/sec ({} messages in {:?})",
            throughput, count, actual_time
        );

        // Should achieve reasonable throughput
        assert!(throughput > 1000.0);

        system.shutdown().await.unwrap();
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_cluster() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        assert_eq!(system.local_actor_names().len(), 0);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_same_actor_name_different_nodes() {
        let config1 = create_cluster_config(20071);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        let mut config2 = create_cluster_config(20072);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Both nodes have actors with same name
        let actor1 = EchoActor::new("shared-name");
        let actor2 = EchoActor::new("shared-name");

        let ref1 = system1.spawn(actor1).await.unwrap();
        let ref2 = system2.spawn(actor2).await.unwrap();

        // They should have different full IDs (different node IDs)
        assert_ne!(ref1.id().node, ref2.id().node);
        assert_eq!(ref1.id().name, ref2.id().name);

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_rapid_spawn_and_stop() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        for i in 0..50 {
            let actor = EchoActor::new(&format!("rapid-{}", i));
            let _ref = system.spawn(actor).await.unwrap();
            system.stop(&format!("rapid-{}", i)).await.unwrap();
        }

        assert_eq!(system.local_actor_names().len(), 0);

        system.shutdown().await.unwrap();
    }
}
