//! Multi-node cluster integration tests

use pulsing_actor::actor::{ActorAddress, ActorId, ActorPath, NodeId};
use pulsing_actor::prelude::*;
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

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Pong {
    result: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Increment;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetCount;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CountResponse {
    count: i32,
}

// ============================================================================
// Test Actors
// ============================================================================

struct Echo;

#[async_trait]
impl Actor for Echo {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        if msg.msg_type().ends_with("Ping") {
            let ping: Ping = msg.unpack()?;
            return Message::pack(&Pong {
                result: ping.value * 2,
            });
        }
        Err(anyhow::anyhow!("Unknown message"))
    }
}

struct Counter {
    count: Arc<AtomicI32>,
}

#[async_trait]
impl Actor for Counter {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        if msg.msg_type().ends_with("Increment") {
            let new_count = self.count.fetch_add(1, Ordering::SeqCst) + 1;
            return Message::pack(&CountResponse { count: new_count });
        }
        if msg.msg_type().ends_with("GetCount") {
            let count = self.count.load(Ordering::SeqCst);
            return Message::pack(&CountResponse { count });
        }
        Err(anyhow::anyhow!("Unknown message"))
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
        let gossip1_addr = system1.addr();

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
        let gossip1_addr = system1.addr();

        // Spawn actor on node 1
        let actor1_ref = system1.spawn("echo", Echo).await.unwrap();

        // Verify local actor works
        let response: Pong = actor1_ref.ask(Ping { value: 21 }).await.unwrap();
        assert_eq!(response.result, 42);

        // Node 2 joins
        let mut config2 = create_cluster_config(20012);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Spawn another actor on node 2
        let actor2_ref = system2.spawn("echo2", Echo).await.unwrap();

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
        let gossip1_addr = system1.addr();

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
        let gossip1_addr = system1.addr();

        let _ref1 = system1.spawn("actor-on-node1", Echo).await.unwrap();

        // Node 2
        let mut config2 = create_cluster_config(20042);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        let _ref2 = system2.spawn("actor-on-node2", Echo).await.unwrap();

        // Node 3
        let mut config3 = create_cluster_config(20043);
        config3.seed_nodes = vec![gossip1_addr];
        let system3 = ActorSystem::new(config3).await.unwrap();

        let _ref3 = system3.spawn("actor-on-node3", Echo).await.unwrap();

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

        let actor = Counter {
            count: count.clone(),
        };
        let actor_ref = system.spawn("counter", actor).await.unwrap();

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

        let actor = Counter {
            count: count.clone(),
        };
        let actor_ref = system.spawn("counter", actor).await.unwrap();

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
        let gossip1_addr = system1.addr();

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
        let gossip1_addr = system1.addr();

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

        let actor_ref = system.spawn("echo", Echo).await.unwrap();

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

        let actor_ref = system.spawn("echo", Echo).await.unwrap();

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
        let gossip1_addr = system1.addr();

        let mut config2 = create_cluster_config(20072);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Both nodes have actors with same name
        let ref1 = system1.spawn("shared-name", Echo).await.unwrap();
        let ref2 = system2.spawn("shared-name", Echo).await.unwrap();

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
            let _ref = system.spawn(format!("rapid-{}", i), Echo).await.unwrap();
            system.stop(&format!("rapid-{}", i)).await.unwrap();
        }

        assert_eq!(system.local_actor_names().len(), 0);

        system.shutdown().await.unwrap();
    }
}

// ============================================================================
// Multi-Node Addressing Tests
// ============================================================================

mod addressing_multi_node_tests {
    use super::*;

    /// Helper to wait for gossip propagation with retry
    async fn wait_for_named_actor<F, T>(
        checker: F,
        max_attempts: u32,
        delay_ms: u64,
        desc: &str,
    ) -> Option<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<T>> + Send>>,
    {
        for attempt in 1..=max_attempts {
            if let Some(result) = checker().await {
                return Some(result);
            }
            if attempt < max_attempts {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
        eprintln!("{} not found after {} attempts", desc, max_attempts);
        None
    }

    #[tokio::test]
    async fn test_named_actor_cluster_registration() {
        // Node 1
        let config1 = create_cluster_config(20081);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.addr();

        // Node 2 joins the cluster
        let mut config2 = create_cluster_config(20082);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Wait for cluster to form
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create named actor on node 1
        let path = ActorPath::new("services/echo").unwrap();
        let _actor_ref = system1
            .spawn_named(path.clone(), "echo_impl", Echo)
            .await
            .unwrap();

        // Wait for gossip to propagate with retries
        let path_clone = path.clone();
        let system2_clone = system2.clone();
        let info = wait_for_named_actor(
            || {
                let p = path_clone.clone();
                let s = system2_clone.clone();
                Box::pin(async move { s.lookup_named(&p).await })
            },
            10,
            200,
            "Named actor",
        )
        .await;

        assert!(
            info.is_some(),
            "Node 2 should see the named actor after gossip"
        );
        let info = info.unwrap();
        assert_eq!(info.instance_count(), 1);

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_remote_named_actor() {
        // Node 1
        let config1 = create_cluster_config(20083);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.addr();

        // Node 2 joins
        let mut config2 = create_cluster_config(20084);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Wait for cluster formation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create named actor on node 1
        let path = ActorPath::new("services/api/handler").unwrap();
        let _actor_ref = system1
            .spawn_named(path.clone(), "api_handler", Echo)
            .await
            .unwrap();

        // Wait for gossip propagation with retries
        let addr = ActorAddress::parse("actor:///services/api/handler").unwrap();
        let mut resolved_ref = None;
        for attempt in 1..=15 {
            match system2.resolve(&addr).await {
                Ok(r) => {
                    resolved_ref = Some(r);
                    break;
                }
                Err(_) if attempt < 15 => {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(e) => {
                    panic!("Failed to resolve after 15 attempts: {}", e);
                }
            }
        }

        let resolved_ref = resolved_ref.expect("Should resolve remote named actor");

        // Should be a remote reference
        assert!(!resolved_ref.is_local());

        // Call should work
        let response: Pong = resolved_ref.ask(Ping { value: 21 }).await.unwrap();
        assert_eq!(response.result, 42);

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_multi_instance_named_actor() {
        // Node 1
        let config1 = create_cluster_config(20085);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.addr();

        // Node 2 joins
        let mut config2 = create_cluster_config(20086);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Wait for cluster formation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create same named actor on BOTH nodes (multi-instance)
        let path = ActorPath::new("services/worker/pool").unwrap();

        let _ref1 = system1
            .spawn_named(path.clone(), "pool_instance_1", Echo)
            .await
            .unwrap();

        // Small delay between registrations
        tokio::time::sleep(Duration::from_millis(100)).await;

        let _ref2 = system2
            .spawn_named(path.clone(), "pool_instance_2", Echo)
            .await
            .unwrap();

        // Wait for gossip propagation with retries until we see 2 instances
        for attempt in 1..=20 {
            let info1 = system1.lookup_named(&path).await;
            let info2 = system2.lookup_named(&path).await;

            let count1 = info1.as_ref().map(|i| i.instance_count()).unwrap_or(0);
            let count2 = info2.as_ref().map(|i| i.instance_count()).unwrap_or(0);

            if count1 >= 2 && count2 >= 2 {
                break;
            }

            if attempt == 20 {
                eprintln!("Instance counts: system1={}, system2={}", count1, count2);
                assert!(
                    count1 >= 2,
                    "System1 should see 2 instances, got {}",
                    count1
                );
                assert!(
                    count2 >= 2,
                    "System2 should see 2 instances, got {}",
                    count2
                );
            }

            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_global_address_cross_node() {
        // Node 1
        let config1 = create_cluster_config(20087);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.addr();
        let node1_id = system1.node_id().clone();

        // Node 2 joins
        let mut config2 = create_cluster_config(20088);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Wait for cluster formation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create regular actor on node 1
        let _actor_ref = system1.spawn("remote_worker", Echo).await.unwrap();

        // Node 2 resolves using global address with retries
        let addr = ActorAddress::global(node1_id.clone(), "remote_worker");
        let mut resolved_ref = None;
        for attempt in 1..=15 {
            match system2.resolve(&addr).await {
                Ok(r) => {
                    resolved_ref = Some(r);
                    break;
                }
                Err(_) if attempt < 15 => {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(e) => {
                    panic!("Failed to resolve global address after 15 attempts: {}", e);
                }
            }
        }

        let resolved_ref = resolved_ref.expect("Should resolve global address");

        // Should be a remote reference
        assert!(!resolved_ref.is_local());

        // Call should work
        let response: Pong = resolved_ref.ask(Ping { value: 10 }).await.unwrap();
        assert_eq!(response.result, 20);

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }
}
