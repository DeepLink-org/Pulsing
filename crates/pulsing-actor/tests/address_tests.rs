//! Comprehensive tests for the actor addressing system

use pulsing_actor::actor::{ActorId, ActorPath};
use pulsing_actor::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ============================================================================
// Test Messages
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EchoMsg {
    value: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EchoResponse {
    value: String,
    from_node: String,
}

// ============================================================================
// Test Actors
// ============================================================================

struct IdentityActor {
    node_name: String,
    call_count: Arc<AtomicUsize>,
}

#[async_trait]
impl Actor for IdentityActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        if msg.msg_type().ends_with("EchoMsg") {
            let echo: EchoMsg = msg.unpack()?;
            self.call_count.fetch_add(1, Ordering::SeqCst);
            return Message::pack(&EchoResponse {
                value: echo.value,
                from_node: self.node_name.clone(),
            });
        }
        Err(anyhow::anyhow!("Unknown message"))
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
    }

    #[test]
    fn test_path_with_special_characters() {
        let path = ActorPath::new("services/llm-model_v2").unwrap();
        assert_eq!(path.name(), "llm-model_v2");

        let path = ActorPath::new("workers/worker_123").unwrap();
        assert_eq!(path.name(), "worker_123");
    }

    #[test]
    fn test_path_child() {
        let path = ActorPath::new("services/base").unwrap();
        let child = path.child("api").unwrap();
        assert_eq!(child.name(), "api");
    }

    #[test]
    fn test_invalid_paths() {
        assert!(ActorPath::new("").is_err());
        // Single segment paths without namespace are invalid
        assert!(ActorPath::new("single").is_err());
    }
}

// ============================================================================
// ActorAddress Tests
// ============================================================================

mod actor_address_tests {
    use super::*;

    #[test]
    fn test_address_parsing() {
        // Test that ActorIds can be created
        let actor_id = ActorId::generate();
        assert_ne!(actor_id.0, 0);

        // Test creating from specific value
        let actor_id2 = ActorId::new(12345);
        assert_eq!(actor_id2.0, 12345);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_named_actor() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor_ref = system
            .spawn_named(
                "services/test/actor",
                IdentityActor {
                    node_name: "local".into(),
                    call_count: counter.clone(),
                },
            )
            .await
            .unwrap();

        let response: EchoResponse = actor_ref
            .ask(EchoMsg {
                value: "hello".into(),
            })
            .await
            .unwrap();

        assert_eq!(response.value, "hello");
        assert_eq!(response.from_node, "local");
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_lookup_by_id() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor_ref = system
            .spawn_named(
                "test/my_actor",
                IdentityActor {
                    node_name: "local".into(),
                    call_count: counter.clone(),
                },
            )
            .await
            .unwrap();

        let actor_id = *actor_ref.id();
        let looked_up_ref = system.actor_ref(&actor_id).await.unwrap();

        let response: EchoResponse = looked_up_ref
            .ask(EchoMsg {
                value: "hello".into(),
            })
            .await
            .unwrap();

        assert_eq!(response.value, "hello");
        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_duplicate_path_registration() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let counter1 = Arc::new(AtomicUsize::new(0));
        let result1 = system
            .spawn_named(
                "services/unique",
                IdentityActor {
                    node_name: "local".into(),
                    call_count: counter1,
                },
            )
            .await;
        assert!(result1.is_ok());

        let counter2 = Arc::new(AtomicUsize::new(0));
        // Trying to spawn with the same name should fail
        let result2 = system
            .spawn_named(
                "services/unique",
                IdentityActor {
                    node_name: "local".into(),
                    call_count: counter2,
                },
            )
            .await;
        assert!(result2.is_err());

        system.shutdown().await.unwrap();
    }
}
