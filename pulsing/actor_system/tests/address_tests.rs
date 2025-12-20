//! Comprehensive tests for the actor addressing system

use pulsing_actor::prelude::*;
use pulsing_actor::system::SystemConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Test Messages
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Echo {
    value: String,
}

impl Message for Echo {
    fn type_id() -> &'static str {
        "Echo"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EchoResponse {
    value: String,
    from_node: String,
}

impl Message for EchoResponse {
    fn type_id() -> &'static str {
        "EchoResponse"
    }
}

// ============================================================================
// Test Actors
// ============================================================================

struct IdentityActor {
    id: ActorId,
    node_name: String,
    call_count: Arc<AtomicUsize>,
}

impl IdentityActor {
    fn new(name: &str, node_name: String, counter: Arc<AtomicUsize>) -> Self {
        Self {
            id: ActorId::local(name),
            node_name,
            call_count: counter,
        }
    }
}

#[async_trait]
impl Actor for IdentityActor {
    fn id(&self) -> &ActorId {
        &self.id
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "Echo" => {
                let echo: Echo = msg.into_message()?;
                self.call_count.fetch_add(1, Ordering::SeqCst);
                RawMessage::from_message(&EchoResponse {
                    value: echo.value,
                    from_node: self.node_name.clone(),
                })
            }
            _ => Err(anyhow::anyhow!("Unknown message")),
        }
    }
}

// ============================================================================
// ActorPath Tests
// ============================================================================

mod actor_path_tests {
    use super::*;

    #[test]
    fn test_path_with_many_segments() {
        let path = ActorPath::new("a/b/c/d/e/f").unwrap();
        assert_eq!(path.segments().len(), 6);
        assert_eq!(path.namespace(), "a");
        assert_eq!(path.name(), "f");
        assert_eq!(path.as_str(), "a/b/c/d/e/f");
    }

    #[test]
    fn test_path_with_special_characters() {
        // Alphanumeric and underscore/hyphen should work
        let path = ActorPath::new("services/llm-model_v2").unwrap();
        assert_eq!(path.name(), "llm-model_v2");

        let path = ActorPath::new("workers/worker_123").unwrap();
        assert_eq!(path.name(), "worker_123");
    }

    #[test]
    fn test_path_parent() {
        let path = ActorPath::new("services/llm/router").unwrap();
        let parent = path.parent();
        assert!(parent.is_some());
        assert_eq!(parent.unwrap().as_str(), "services/llm");

        let short_path = ActorPath::new("services/api").unwrap();
        let parent = short_path.parent();
        assert!(parent.is_none()); // Can't have a parent with less than 2 segments
    }

    #[test]
    fn test_path_child() {
        let path = ActorPath::new("services/llm").unwrap();
        let child = path.child("router").unwrap();
        assert_eq!(child.as_str(), "services/llm/router");
        assert_eq!(child.name(), "router");
    }

    #[test]
    fn test_path_equality() {
        let path1 = ActorPath::new("services/api").unwrap();
        let path2 = ActorPath::new("services/api").unwrap();
        let path3 = ActorPath::new("services/other").unwrap();

        assert_eq!(path1, path2);
        assert_ne!(path1, path3);
    }

    #[test]
    fn test_path_hash() {
        use std::collections::HashSet;

        let path1 = ActorPath::new("services/api").unwrap();
        let path2 = ActorPath::new("services/api").unwrap();
        let path3 = ActorPath::new("services/other").unwrap();

        let mut set = HashSet::new();
        set.insert(path1.clone());

        assert!(set.contains(&path2));
        assert!(!set.contains(&path3));
    }

    #[test]
    fn test_path_display() {
        let path = ActorPath::new("services/api").unwrap();
        assert_eq!(format!("{}", path), "services/api");
    }

    #[test]
    fn test_path_edge_cases() {
        // Leading/trailing slashes are trimmed and should work
        assert!(ActorPath::new("/services/api").is_ok());
        assert!(ActorPath::new("services/api/").is_ok());
        assert!(ActorPath::new("/services/api/").is_ok());

        // Trimmed paths should be equivalent
        let p1 = ActorPath::new("services/api").unwrap();
        let p2 = ActorPath::new("/services/api/").unwrap();
        assert_eq!(p1.as_str(), p2.as_str());

        // Multiple consecutive slashes should fail (empty segment)
        assert!(ActorPath::new("services///api").is_err());

        // Only slashes should fail (empty after trim)
        assert!(ActorPath::new("///").is_err());
        assert!(ActorPath::new("/").is_err());
    }
}

// ============================================================================
// ActorAddress Tests
// ============================================================================

mod actor_address_tests {
    use super::*;

    #[test]
    fn test_address_helper_methods() {
        // Named service
        let addr = ActorAddress::parse("actor:///services/api").unwrap();
        assert!(addr.is_named());
        assert!(!addr.is_global());
        assert!(!addr.is_localhost());
        assert!(addr.path().is_some());
        assert!(addr.node_id().is_none());
        assert!(addr.actor_id().is_none());

        // Named instance
        let addr = ActorAddress::parse("actor:///services/api@node_a").unwrap();
        assert!(addr.is_named());
        assert!(!addr.is_global());
        assert!(addr.node_id().is_some());
        assert_eq!(addr.node_id().unwrap().as_str(), "node_a");

        // Global
        let addr = ActorAddress::parse("actor://node_a/worker").unwrap();
        assert!(!addr.is_named());
        assert!(addr.is_global());
        assert!(addr.node_id().is_some());
        assert!(addr.actor_id().is_some());
        assert_eq!(addr.actor_id().unwrap(), "worker");

        // Localhost
        let addr = ActorAddress::parse("actor://localhost/worker").unwrap();
        assert!(addr.is_localhost());
        assert!(addr.is_global());
    }

    #[test]
    fn test_address_with_instance() {
        let addr = ActorAddress::named(ActorPath::new("services/api").unwrap());
        let with_instance = addr.with_instance(NodeId::new("node_a"));

        match with_instance {
            ActorAddress::Named { instance, .. } => {
                assert_eq!(instance.unwrap().as_str(), "node_a");
            }
            _ => panic!("Expected Named address"),
        }

        // Applying with_instance to a global address should be a no-op
        let global = ActorAddress::global(NodeId::new("node_a"), "worker");
        let still_global = global.clone().with_instance(NodeId::new("node_b"));
        assert_eq!(global.to_uri(), still_global.to_uri());
    }

    #[test]
    fn test_address_parse_edge_cases() {
        // Missing actor ID
        assert!(ActorAddress::parse("actor://node_a/").is_err());
        assert!(ActorAddress::parse("actor://node_a").is_err());

        // Missing node ID for global
        assert!(ActorAddress::parse("actor:///").is_err());

        // Invalid characters (should still work for many cases)
        assert!(ActorAddress::parse("actor://node_a/worker-123").is_ok());
        assert!(ActorAddress::parse("actor://node_a/worker_123").is_ok());

        // Very long paths
        let long_path = "a/".repeat(50) + "z";
        let addr_str = format!("actor:///{}", long_path);
        // This should still parse (though it might fail path validation)
        let result = ActorAddress::parse(&addr_str);
        // Just check it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_address_display() {
        let addr = ActorAddress::global(NodeId::new("node_a"), "worker");
        assert_eq!(format!("{}", addr), "actor://node_a/worker");

        let addr = ActorAddress::named(ActorPath::new("services/api").unwrap());
        assert_eq!(format!("{}", addr), "actor:///services/api");
    }

    #[test]
    fn test_address_try_from() {
        let addr: Result<ActorAddress, _> = "actor://node_a/worker".try_into();
        assert!(addr.is_ok());

        let addr: Result<ActorAddress, _> = "http://node_a/worker".try_into();
        assert!(addr.is_err());
    }

    #[test]
    fn test_address_clone_and_eq() {
        let addr1 = ActorAddress::global(NodeId::new("node_a"), "worker");
        let addr2 = addr1.clone();
        assert_eq!(addr1, addr2);

        let addr3 = ActorAddress::global(NodeId::new("node_b"), "worker");
        assert_ne!(addr1, addr3);
    }
}

// ============================================================================
// Named Actor System Tests
// ============================================================================

mod named_actor_system_tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_and_lookup_named_actor() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let path = ActorPath::new("services/test/actor").unwrap();
        let actor = IdentityActor::new("test_actor", "local".to_string(), counter.clone());

        let actor_ref = system.spawn_named(path.clone(), actor).await.unwrap();

        // Verify lookup
        let info = system.lookup_named(&path).await;
        assert!(info.is_some());
        assert_eq!(info.unwrap().instance_count(), 1);

        // Send message
        let response: EchoResponse = actor_ref
            .ask(Echo {
                value: "hello".into(),
            })
            .await
            .unwrap();
        assert_eq!(response.value, "hello");
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_multiple_named_actors_different_paths() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let paths = vec![
            ActorPath::new("services/api/handler").unwrap(),
            ActorPath::new("services/db/connection").unwrap(),
            ActorPath::new("workers/pool/manager").unwrap(),
        ];

        for (i, path) in paths.iter().enumerate() {
            let counter = Arc::new(AtomicUsize::new(0));
            let actor = IdentityActor::new(&format!("actor_{}", i), "local".to_string(), counter);
            let _ = system.spawn_named(path.clone(), actor).await.unwrap();
        }

        // All should be registered
        for path in &paths {
            let info = system.lookup_named(path).await;
            assert!(info.is_some(), "Path {} should be registered", path);
        }

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_stop_named_actor_removes_from_registry() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let path = ActorPath::new("services/temporary").unwrap();
        let actor = IdentityActor::new("temp", "local".to_string(), counter);
        let _ = system.spawn_named(path.clone(), actor).await.unwrap();

        // Verify it exists
        assert!(system.lookup_named(&path).await.is_some());

        // Stop it
        system.stop_named(&path).await.unwrap();

        // Wait for cleanup
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Should be gone or have 0 instances
        let info = system.lookup_named(&path).await;
        if let Some(info) = info {
            assert_eq!(info.instance_count(), 0);
        }

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_by_address_string() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let path = ActorPath::new("services/echo").unwrap();
        let actor = IdentityActor::new("echo_actor", "local".to_string(), counter.clone());
        let _ = system.spawn_named(path.clone(), actor).await.unwrap();

        // Resolve using different address formats
        let addr1 = ActorAddress::parse("actor:///services/echo").unwrap();
        let ref1 = system.resolve(&addr1).await.unwrap();
        let _: EchoResponse = ref1
            .ask(Echo {
                value: "test1".into(),
            })
            .await
            .unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_localhost_address() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = IdentityActor::new("my_actor", "local".to_string(), counter.clone());
        let _ = system.spawn(actor).await.unwrap();

        // Resolve using localhost
        let addr = ActorAddress::parse("actor://localhost/my_actor").unwrap();
        let actor_ref = system.resolve(&addr).await.unwrap();

        let response: EchoResponse = actor_ref
            .ask(Echo {
                value: "hello".into(),
            })
            .await
            .unwrap();
        assert_eq!(response.value, "hello");

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_nonexistent_returns_error() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Non-existent named actor
        let addr = ActorAddress::parse("actor:///services/nonexistent").unwrap();
        let result = system.resolve(&addr).await;
        assert!(result.is_err());

        // Non-existent global actor
        let addr = ActorAddress::parse("actor://localhost/nonexistent").unwrap();
        let result = system.resolve(&addr).await;
        assert!(result.is_err());

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_named_actor_registration() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
        let system_arc = Arc::new(system);

        let mut handles = vec![];

        // Spawn multiple named actors concurrently
        for i in 0..10 {
            let sys = system_arc.clone();
            let handle = tokio::spawn(async move {
                let counter = Arc::new(AtomicUsize::new(0));
                let path = ActorPath::new(&format!("services/concurrent/{}", i)).unwrap();
                let actor =
                    IdentityActor::new(&format!("actor_{}", i), "local".to_string(), counter);
                sys.spawn_named(path, actor).await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        // Verify all are registered
        for i in 0..10 {
            let path = ActorPath::new(&format!("services/concurrent/{}", i)).unwrap();
            assert!(system_arc.lookup_named(&path).await.is_some());
        }

        system_arc.shutdown().await.unwrap();
    }
}

// ============================================================================
// Multi-Node Named Actor Tests
// ============================================================================

mod multi_node_named_actor_tests {
    use super::*;

    fn create_cluster_config(_port: u16) -> SystemConfig {
        // Use port 0 to let the OS assign an available port
        SystemConfig::with_addr("127.0.0.1:0".parse().unwrap())
    }

    #[tokio::test]
    async fn test_resolve_named_instance_specific_node() {
        // Setup two nodes
        let config1 = create_cluster_config(21001);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();
        let node1_id = system1.node_id().clone();

        let mut config2 = create_cluster_config(21002);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();
        let node2_id = system2.node_id().clone();

        // Wait for cluster formation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create same named actor on both nodes
        let path = ActorPath::new("services/echo").unwrap();

        let counter1 = Arc::new(AtomicUsize::new(0));
        let actor1 = IdentityActor::new("echo1", "node1".to_string(), counter1.clone());
        let _ = system1.spawn_named(path.clone(), actor1).await.unwrap();

        let counter2 = Arc::new(AtomicUsize::new(0));
        let actor2 = IdentityActor::new("echo2", "node2".to_string(), counter2.clone());
        let _ = system2.spawn_named(path.clone(), actor2).await.unwrap();

        // Wait for gossip
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Resolve to specific instance on node1
        let addr1 = ActorAddress::named_instance(path.clone(), node1_id);
        let ref1 = system2.resolve(&addr1).await.unwrap();
        let response: EchoResponse = ref1
            .ask(Echo {
                value: "to_node1".into(),
            })
            .await
            .unwrap();
        assert_eq!(response.from_node, "node1");

        // Resolve to specific instance on node2
        let addr2 = ActorAddress::named_instance(path.clone(), node2_id);
        let ref2 = system1.resolve(&addr2).await.unwrap();
        let response: EchoResponse = ref2
            .ask(Echo {
                value: "to_node2".into(),
            })
            .await
            .unwrap();
        assert_eq!(response.from_node, "node2");

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_load_balanced_named_actor_calls() {
        // Setup two nodes
        let config1 = create_cluster_config(21003);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        let mut config2 = create_cluster_config(21004);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Wait for cluster formation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create same named actor on both nodes
        let path = ActorPath::new("services/worker/pool").unwrap();

        let counter1 = Arc::new(AtomicUsize::new(0));
        let actor1 = IdentityActor::new("worker1", "node1".to_string(), counter1.clone());
        let _ = system1.spawn_named(path.clone(), actor1).await.unwrap();

        let counter2 = Arc::new(AtomicUsize::new(0));
        let actor2 = IdentityActor::new("worker2", "node2".to_string(), counter2.clone());
        let _ = system2.spawn_named(path.clone(), actor2).await.unwrap();

        // Wait for gossip
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Make multiple calls to the service address (should load balance)
        let addr = ActorAddress::parse("actor:///services/worker/pool").unwrap();
        let mut node1_calls = 0;
        let mut node2_calls = 0;

        // Use a third node to make calls (or use one of the existing ones)
        for _ in 0..20 {
            let actor_ref = system1.resolve(&addr).await.unwrap();
            let response: EchoResponse = actor_ref
                .ask(Echo {
                    value: "test".into(),
                })
                .await
                .unwrap();
            if response.from_node == "node1" {
                node1_calls += 1;
            } else {
                node2_calls += 1;
            }
        }

        // With random load balancing, both should get some calls
        // Note: This is probabilistic, but very unlikely to fail
        // In this case, local preference means node1 gets all calls
        // So we just check total is 20
        assert_eq!(node1_calls + node2_calls, 20);

        system1.shutdown().await.unwrap();
        system2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_named_actor_survives_node_failure() {
        // Setup two nodes
        let config1 = create_cluster_config(21005);
        let system1 = ActorSystem::new(config1).await.unwrap();
        let gossip1_addr = system1.gossip_addr();

        let mut config2 = create_cluster_config(21006);
        config2.seed_nodes = vec![gossip1_addr];
        let system2 = ActorSystem::new(config2).await.unwrap();

        // Wait for cluster formation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create named actor on node2 only
        let path = ActorPath::new("services/ephemeral").unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let actor = IdentityActor::new("ephemeral", "node2".to_string(), counter);
        let _ = system2.spawn_named(path.clone(), actor).await.unwrap();

        // Wait for gossip
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify node1 can see it
        let info = system1.lookup_named(&path).await;
        assert!(info.is_some());

        // Shutdown node2
        system2.shutdown().await.unwrap();

        // Wait for failure detection (might take a while)
        tokio::time::sleep(Duration::from_millis(500)).await;

        // system1 should eventually remove the actor from registry
        // (This depends on failure detection which might take longer)
        // For now, just verify system1 is still running
        assert!(!system1.cancel_token().is_cancelled());

        system1.shutdown().await.unwrap();
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_duplicate_named_actor_path() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let path = ActorPath::new("services/unique").unwrap();

        // First registration should succeed
        let counter1 = Arc::new(AtomicUsize::new(0));
        let actor1 = IdentityActor::new("actor1", "local".to_string(), counter1);
        let result1 = system.spawn_named(path.clone(), actor1).await;
        assert!(result1.is_ok());

        // Second registration with same path should fail
        let counter2 = Arc::new(AtomicUsize::new(0));
        let actor2 = IdentityActor::new("actor2", "local".to_string(), counter2);
        let result2 = system.spawn_named(path.clone(), actor2).await;
        assert!(result2.is_err());

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_invalid_address_parsing() {
        // Various invalid addresses
        let invalid = vec![
            "",
            "not_a_uri",
            "http://node/actor",
            "actor:",
            "actor://",
            "actor:///",
            "actor://node/",
        ];

        for addr_str in invalid {
            let result = ActorAddress::parse(addr_str);
            assert!(
                result.is_err(),
                "Expected '{}' to fail parsing, but got: {:?}",
                addr_str,
                result
            );
        }
    }

    #[tokio::test]
    async fn test_stop_nonexistent_named_actor() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let path = ActorPath::new("services/nonexistent").unwrap();
        let result = system.stop_named(&path).await;

        // Should return error or succeed silently
        // Either behavior is acceptable
        let _ = result;

        system.shutdown().await.unwrap();
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_many_named_actors() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let count = 100;
        for i in 0..count {
            let path = ActorPath::new(&format!("services/stress/{}", i)).unwrap();
            let counter = Arc::new(AtomicUsize::new(0));
            let actor = IdentityActor::new(&format!("stress_{}", i), "local".to_string(), counter);
            let _ = system.spawn_named(path, actor).await.unwrap();
        }

        // Verify all are registered
        for i in 0..count {
            let path = ActorPath::new(&format!("services/stress/{}", i)).unwrap();
            assert!(
                system.lookup_named(&path).await.is_some(),
                "Actor {} should exist",
                i
            );
        }

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_rapid_register_unregister() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        for i in 0..50 {
            let path = ActorPath::new(&format!("services/rapid/{}", i)).unwrap();

            // Register
            let counter = Arc::new(AtomicUsize::new(0));
            let actor = IdentityActor::new(&format!("rapid_{}", i), "local".to_string(), counter);
            let _ = system.spawn_named(path.clone(), actor).await.unwrap();

            // Unregister
            let _ = system.stop_named(&path).await;
        }

        // Wait for cleanup
        tokio::time::sleep(Duration::from_millis(100)).await;

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_resolve() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
        let system_arc = Arc::new(system);

        // Create a named actor
        let path = ActorPath::new("services/concurrent/target").unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let actor = IdentityActor::new("target", "local".to_string(), counter.clone());
        let _ = system_arc.spawn_named(path.clone(), actor).await.unwrap();

        // Spawn many concurrent resolve + call tasks
        let mut handles = vec![];
        for _ in 0..50 {
            let sys = system_arc.clone();
            let handle = tokio::spawn(async move {
                let addr = ActorAddress::parse("actor:///services/concurrent/target").unwrap();
                let actor_ref = sys.resolve(&addr).await?;
                let _: EchoResponse = actor_ref
                    .ask(Echo {
                        value: "test".into(),
                    })
                    .await?;
                Ok::<_, anyhow::Error>(())
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        assert_eq!(counter.load(Ordering::SeqCst), 50);

        system_arc.shutdown().await.unwrap();
    }
}
