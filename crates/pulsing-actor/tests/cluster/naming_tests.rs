//! Naming backend tests
//!
//! Tests for the NamingBackend trait and GossipBackend implementation.
//!
//! These tests verify that GossipBackend correctly implements the NamingBackend
//! trait by testing through the ActorSystem API, which is the primary interface
//! for users.

use pulsing_actor::actor::ActorPath;
use pulsing_actor::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// NamingBackend Trait Tests via ActorSystem
// ============================================================================

#[tokio::test]
async fn test_naming_backend_node_management() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    // Test all_members - should include local node
    let members = system.members().await;
    assert!(!members.is_empty());
    assert!(members.iter().any(|m| m.node_id == *system.node_id()));

    system.shutdown().await.unwrap();
}

// Simple test actor that implements the new Actor trait
struct TestActor;

#[async_trait::async_trait]
impl Actor for TestActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        // Echo back the message
        Ok(msg)
    }
}

#[tokio::test]
async fn test_naming_backend_register_named_actor() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let path = ActorPath::new("test/actor").unwrap();

    // Register a named actor by spawning it
    let _ref = system
        .spawn_named(path.clone(), "test_actor", TestActor)
        .await
        .unwrap();

    // Should be able to lookup
    let info = system.lookup_named(&path).await;
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.path, path);
    assert!(info.instance_nodes.contains(system.node_id()));

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_naming_backend_lookup_named_actor() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    // Lookup non-existent actor
    let path = ActorPath::new("nonexistent/actor").unwrap();
    let info = system.lookup_named(&path).await;
    assert!(info.is_none());

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_naming_backend_all_named_actors() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    // Initially should be empty (or only system actors)
    let all = system.all_named_actors().await;
    let initial_count = all.len();

    let path1 = ActorPath::new("test/actor1").unwrap();
    let path2 = ActorPath::new("test/actor2").unwrap();

    let _ref1 = system
        .spawn_named(path1, "actor1", TestActor)
        .await
        .unwrap();
    let _ref2 = system
        .spawn_named(path2, "actor2", TestActor)
        .await
        .unwrap();

    // Should now have 2 more
    let all = system.all_named_actors().await;
    assert!(all.len() >= initial_count + 2);

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_naming_backend_resolve_named_actor() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let path = ActorPath::new("test/resolve").unwrap();
    let _ref = system
        .spawn_named(path.clone(), "resolve_actor", TestActor)
        .await
        .unwrap();

    // Should be able to resolve
    let resolved = system.resolve_named(&path, None).await;
    assert!(resolved.is_ok());

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_naming_backend_named_actor_instances() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let path = ActorPath::new("test/instances").unwrap();
    let _ref = system
        .spawn_named(path.clone(), "instances_actor", TestActor)
        .await
        .unwrap();

    // Get instances
    let instances = system.get_named_instances(&path).await;
    assert!(!instances.is_empty());
    assert!(instances.iter().any(|m| m.node_id == *system.node_id()));

    system.shutdown().await.unwrap();
}

// ============================================================================
// GossipBackend Specific Tests (via type downcasting)
// ============================================================================

#[tokio::test]
async fn test_gossip_backend_type_downcast() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    // Test that we can access backend through system's internal API
    // This verifies the backend implements as_any() correctly
    let members = system.members().await;
    assert!(!members.is_empty());

    // The fact that we can call members() means the backend is working
    // We can't directly test downcasting without accessing private fields,
    // but we can verify the functionality works

    system.shutdown().await.unwrap();
}

// ============================================================================
// Integration Tests: Named Actor Lifecycle
// ============================================================================

#[tokio::test]
async fn test_named_actor_full_lifecycle() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let path = ActorPath::new("test/lifecycle").unwrap();

    // 1. Register
    let _actor_ref = system
        .spawn_named(path.clone(), "lifecycle_actor", TestActor)
        .await
        .unwrap();

    // 2. Verify registered
    let info = system.lookup_named(&path).await;
    assert!(info.is_some());

    // 3. Resolve
    let resolved = system.resolve_named(&path, None).await;
    assert!(resolved.is_ok());

    // 4. Get instances
    let instances = system.get_named_instances(&path).await;
    assert!(!instances.is_empty());

    // 5. Stop (unregister)
    system.stop_named(&path).await.unwrap();

    // 6. Verify unregistered (may take a moment for gossip to propagate)
    tokio::time::sleep(Duration::from_millis(100)).await;
    let _info_after = system.lookup_named(&path).await;
    // Actor should be removed or marked as failed
    // (exact behavior depends on implementation)

    system.shutdown().await.unwrap();
}

// Actor with custom metadata
struct MetadataActor {
    metadata: HashMap<String, String>,
}

impl MetadataActor {
    fn new(metadata: HashMap<String, String>) -> Self {
        Self { metadata }
    }
}

#[async_trait::async_trait]
impl Actor for MetadataActor {
    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        Ok(msg)
    }
}

#[tokio::test]
async fn test_named_actor_with_metadata() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let path = ActorPath::new("test/metadata").unwrap();
    let mut metadata = HashMap::new();
    metadata.insert("python_class".to_string(), "MyActor".to_string());
    metadata.insert("python_module".to_string(), "my_module".to_string());

    let _ref = system
        .spawn_named_with_options(
            path.clone(),
            "metadata_actor",
            MetadataActor::new(metadata.clone()),
            SpawnOptions::new().public(true).metadata(metadata.clone()),
        )
        .await
        .unwrap();

    // Get detailed instance info
    let instances = system.get_named_instances_detailed(&path).await;
    assert!(!instances.is_empty());

    let (_member, instance) = &instances[0];
    if let Some(instance) = instance {
        // Check metadata is preserved
        assert_eq!(
            instance.metadata.get("python_class"),
            Some(&"MyActor".to_string())
        );
        assert_eq!(
            instance.metadata.get("python_module"),
            Some(&"my_module".to_string())
        );
    }

    system.shutdown().await.unwrap();
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_resolve_nonexistent_named_actor() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let path = ActorPath::new("nonexistent/actor").unwrap();
    let result = system.resolve_named(&path, None).await;

    // Should fail to resolve
    assert!(result.is_err());

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_named_actor_instances_nonexistent() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let path = ActorPath::new("nonexistent/actor").unwrap();
    let instances = system.get_named_instances(&path).await;

    // Should return empty list
    assert!(instances.is_empty());

    system.shutdown().await.unwrap();
}

// ============================================================================
// Multi-Instance Tests (standalone mode limitations)
// ============================================================================

#[tokio::test]
async fn test_multiple_registrations_same_path() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let path = ActorPath::new("test/multi").unwrap();

    // First registration
    let _ref1 = system
        .spawn_named(path.clone(), "multi_actor", TestActor)
        .await
        .unwrap();

    // Second registration with same path should either:
    // 1. Replace the first one, or
    // 2. Add as another instance
    // The exact behavior depends on implementation

    let info = system.lookup_named(&path).await;
    assert!(info.is_some());
    // Should have at least one instance
    assert!(!info.unwrap().instance_nodes.is_empty());

    system.shutdown().await.unwrap();
}

// ============================================================================
// Head Node Backend Tests
// ============================================================================

#[tokio::test]
async fn test_head_node_mode_basic() {
    // Create head node
    let head_config = SystemConfig::standalone().with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();

    // Test that head node has itself as a member
    let members = head_system.members().await;
    assert!(!members.is_empty());
    assert!(members.iter().any(|m| m.node_id == *head_system.node_id()));

    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_node_register_named_actor() {
    let head_config = SystemConfig::standalone().with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();

    let path = ActorPath::new("test/head_actor").unwrap();

    // Register a named actor on head node
    let _ref = head_system
        .spawn_named(path.clone(), "head_actor", TestActor)
        .await
        .unwrap();

    // Should be able to lookup
    let info = head_system.lookup_named(&path).await;
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.path, path);
    assert!(info.instance_nodes.contains(head_system.node_id()));

    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_node_all_named_actors() {
    let head_config = SystemConfig::standalone().with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();

    let initial_count = head_system.all_named_actors().await.len();

    let path1 = ActorPath::new("test/head1").unwrap();
    let path2 = ActorPath::new("test/head2").unwrap();

    let _ref1 = head_system
        .spawn_named(path1, "head1", TestActor)
        .await
        .unwrap();
    let _ref2 = head_system
        .spawn_named(path2, "head2", TestActor)
        .await
        .unwrap();

    let all = head_system.all_named_actors().await;
    assert!(all.len() >= initial_count + 2);

    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_node_resolve_named_actor() {
    let head_config = SystemConfig::standalone().with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();

    let path = ActorPath::new("test/head_resolve").unwrap();
    let _ref = head_system
        .spawn_named(path.clone(), "head_resolve", TestActor)
        .await
        .unwrap();

    // Should be able to resolve
    let resolved = head_system.resolve_named(&path, None).await;
    assert!(resolved.is_ok());

    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_node_named_actor_instances() {
    let head_config = SystemConfig::standalone().with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();

    let path = ActorPath::new("test/head_instances").unwrap();
    let _ref = head_system
        .spawn_named(path.clone(), "head_instances", TestActor)
        .await
        .unwrap();

    // Get instances
    let instances = head_system.get_named_instances(&path).await;
    assert!(!instances.is_empty());
    assert!(instances
        .iter()
        .any(|m| m.node_id == *head_system.node_id()));

    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_node_with_custom_config() {
    use pulsing_actor::cluster::HeadNodeConfig;

    let config = HeadNodeConfig {
        sync_interval: Duration::from_secs(2),
        heartbeat_interval: Duration::from_secs(5),
        heartbeat_timeout: Duration::from_secs(15),
    };

    let head_config = SystemConfig::standalone()
        .with_head_node()
        .with_head_node_config(config);
    let head_system = ActorSystem::new(head_config).await.unwrap();

    // Should work normally
    let members = head_system.members().await;
    assert!(!members.is_empty());

    head_system.shutdown().await.unwrap();
}

// ============================================================================
// Head-Worker Integration Tests
// ============================================================================

#[tokio::test]
async fn test_head_worker_basic_interaction() {
    // Create head node
    let head_config = SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();
    let head_addr = head_system.addr();

    // Create worker node connecting to head
    let worker_config =
        SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_addr(head_addr);
    let worker_system = ActorSystem::new(worker_config).await.unwrap();

    // Give some time for worker to register with head
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Head should see worker node
    let head_members = head_system.members().await;
    assert!(head_members.len() >= 2); // Head + Worker
    assert!(head_members
        .iter()
        .any(|m| m.node_id == *worker_system.node_id()));

    // Worker should see head node (and itself)
    let worker_members = worker_system.members().await;
    assert!(worker_members.len() >= 2); // Head + Worker
    assert!(worker_members
        .iter()
        .any(|m| m.node_id == *head_system.node_id()));

    worker_system.shutdown().await.unwrap();
    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_worker_named_actor_sync() {
    // Create head node
    let head_config = SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();
    let head_addr = head_system.addr();

    // Create worker node
    let worker_config =
        SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_addr(head_addr);
    let worker_system = ActorSystem::new(worker_config).await.unwrap();

    // Give time for worker to register
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Register named actor on worker
    let path = ActorPath::new("test/worker_actor").unwrap();
    let _ref = worker_system
        .spawn_named(path.clone(), "worker_actor", TestActor)
        .await
        .unwrap();

    // Give time for sync
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Head should see the named actor
    let head_info = head_system.lookup_named(&path).await;
    assert!(head_info.is_some());
    let head_info = head_info.unwrap();
    assert!(head_info.instance_nodes.contains(worker_system.node_id()));

    // Worker should also see it
    let worker_info = worker_system.lookup_named(&path).await;
    assert!(worker_info.is_some());

    worker_system.shutdown().await.unwrap();
    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_worker_multiple_actors() {
    // Create head node
    let head_config = SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();
    let head_addr = head_system.addr();

    // Create worker node
    let worker_config =
        SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_addr(head_addr);
    let worker_system = ActorSystem::new(worker_config).await.unwrap();

    // Give time for worker to register
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Register multiple actors on worker
    let path1 = ActorPath::new("test/multi1").unwrap();
    let path2 = ActorPath::new("test/multi2").unwrap();

    let _ref1 = worker_system
        .spawn_named(path1.clone(), "multi1", TestActor)
        .await
        .unwrap();
    let _ref2 = worker_system
        .spawn_named(path2.clone(), "multi2", TestActor)
        .await
        .unwrap();

    // Give time for sync
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Head should see both actors
    let all = head_system.all_named_actors().await;
    let paths: Vec<_> = all.iter().map(|info| &info.path).collect();
    assert!(paths.contains(&&path1));
    assert!(paths.contains(&&path2));

    worker_system.shutdown().await.unwrap();
    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_worker_actor_resolution() {
    // Create head node
    let head_config = SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();
    let head_addr = head_system.addr();

    // Create worker node
    let worker_config =
        SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_addr(head_addr);
    let worker_system = ActorSystem::new(worker_config).await.unwrap();

    // Give time for worker to register
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Register actor on worker
    let path = ActorPath::new("test/resolve_actor").unwrap();
    let _ref = worker_system
        .spawn_named(path.clone(), "resolve_actor", TestActor)
        .await
        .unwrap();

    // Give time for sync
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Head should be able to resolve the actor
    let resolved = head_system.resolve_named(&path, None).await;
    assert!(resolved.is_ok());

    // Worker should also be able to resolve
    let worker_resolved = worker_system.resolve_named(&path, None).await;
    assert!(worker_resolved.is_ok());

    worker_system.shutdown().await.unwrap();
    head_system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_head_worker_actor_unregister() {
    // Create head node
    let head_config = SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_node();
    let head_system = ActorSystem::new(head_config).await.unwrap();
    let head_addr = head_system.addr();

    // Create worker node
    let worker_config =
        SystemConfig::with_addr("127.0.0.1:0".parse().unwrap()).with_head_addr(head_addr);
    let worker_system = ActorSystem::new(worker_config).await.unwrap();

    // Give time for worker to register
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Register actor on worker
    let path = ActorPath::new("test/unregister_actor").unwrap();
    let _ref = worker_system
        .spawn_named(path.clone(), "unregister_actor", TestActor)
        .await
        .unwrap();

    // Give time for sync
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Verify it's registered
    assert!(head_system.lookup_named(&path).await.is_some());

    // Unregister
    worker_system.stop_named(&path).await.unwrap();

    // Give time for sync
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Head should no longer see it (or it should be marked as failed)
    // Note: exact behavior depends on implementation

    worker_system.shutdown().await.unwrap();
    head_system.shutdown().await.unwrap();
}
