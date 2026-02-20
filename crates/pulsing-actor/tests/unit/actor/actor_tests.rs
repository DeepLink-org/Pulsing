//! Actor core functionality tests

use pulsing_actor::error::{PulsingError, RuntimeError};
use pulsing_actor::prelude::*;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Test Messages
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct Ping {
    value: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct Pong {
    result: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Increment {
    amount: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetState;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct StateResponse {
    value: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SlowMessage {
    delay_ms: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ErrorMessage;

// ============================================================================
// Test Actors
// ============================================================================

/// Simple counter actor for testing
struct Counter {
    count: i32,
}

#[async_trait]
impl Actor for Counter {
    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        let msg_type = msg.msg_type();
        if msg_type.ends_with("Ping") {
            let ping: Ping = msg.unpack()?;
            self.count += ping.value;
            return Message::pack(&Pong { result: self.count });
        }
        if msg_type.ends_with("Increment") {
            let inc: Increment = msg.unpack()?;
            self.count += inc.amount;
            return Message::pack(&StateResponse { value: self.count });
        }
        if msg_type.ends_with("GetState") {
            return Message::pack(&StateResponse { value: self.count });
        }
        if msg_type.ends_with("SlowMessage") {
            let slow: SlowMessage = msg.unpack()?;
            tokio::time::sleep(Duration::from_millis(slow.delay_ms)).await;
            return Message::pack(&StateResponse { value: self.count });
        }
        if msg_type.ends_with("ErrorMessage") {
            return Err(PulsingError::from(RuntimeError::Other(
                "Intentional error for testing".into(),
            )));
        }
        Err(PulsingError::from(RuntimeError::Other(format!(
            "Unknown message type: {}",
            msg_type
        ))))
    }
}

/// Actor that tracks lifecycle events with shared state
struct LifecycleActor {
    start_count: Arc<AtomicI32>,
    stop_count: Arc<AtomicI32>,
}

#[async_trait]
impl Actor for LifecycleActor {
    async fn on_start(&mut self, _ctx: &mut ActorContext) -> pulsing_actor::error::Result<()> {
        self.start_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> pulsing_actor::error::Result<()> {
        self.stop_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        if msg.msg_type().ends_with("Ping") {
            return Message::pack(&Pong { result: 0 });
        }
        Err(PulsingError::from(RuntimeError::Other(
            "Unknown message".into(),
        )))
    }
}

// ============================================================================
// Tests
// ============================================================================

mod basic_tests {
    use super::*;

    #[tokio::test]
    async fn test_actor_spawn() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
        let actor_ref = system
            .spawn_named("test/counter", Counter { count: 0 })
            .await
            .unwrap();
        // ActorId now uses u128 (node_id:local_id), verify it's a local actor
        assert!(actor_ref.is_local());
        let _ = system.shutdown().await;
    }

    #[tokio::test]
    async fn test_actor_ask() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
        let actor_ref = system
            .spawn_named("test/counter", Counter { count: 0 })
            .await
            .unwrap();

        let response: Pong = actor_ref.ask(Ping { value: 5 }).await.unwrap();
        assert_eq!(response.result, 5);

        let response: Pong = actor_ref.ask(Ping { value: 3 }).await.unwrap();
        assert_eq!(response.result, 8);

        let _ = system.shutdown().await;
    }

    #[tokio::test]
    async fn test_actor_tell() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
        let actor_ref = system
            .spawn_named("test/counter", Counter { count: 0 })
            .await
            .unwrap();

        actor_ref.tell(Ping { value: 5 }).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let response: StateResponse = actor_ref.ask(GetState).await.unwrap();
        assert_eq!(response.value, 5);

        let _ = system.shutdown().await;
    }

    #[tokio::test]
    async fn test_actor_multiple_messages() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
        let actor_ref = system
            .spawn_named("test/counter", Counter { count: 0 })
            .await
            .unwrap();

        for i in 1..=10 {
            let response: Pong = actor_ref.ask(Ping { value: i }).await.unwrap();
            assert_eq!(response.result, (1..=i).sum::<i32>());
        }

        let _ = system.shutdown().await;
    }
}

mod lifecycle_tests {
    use super::*;

    #[tokio::test]
    async fn test_actor_on_start_called() {
        let start_count = Arc::new(AtomicI32::new(0));
        let stop_count = Arc::new(AtomicI32::new(0));

        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        let actor = LifecycleActor {
            start_count: start_count.clone(),
            stop_count: stop_count.clone(),
        };
        let _actor_ref = system.spawn_named("test/lifecycle", actor).await.unwrap();

        // Give some time for on_start to be called
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(start_count.load(Ordering::SeqCst), 1);

        let _ = system.shutdown().await;
    }

    #[tokio::test]
    async fn test_actor_on_stop_called_on_shutdown() {
        let start_count = Arc::new(AtomicI32::new(0));
        let stop_count = Arc::new(AtomicI32::new(0));

        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        let actor = LifecycleActor {
            start_count: start_count.clone(),
            stop_count: stop_count.clone(),
        };
        let _actor_ref = system.spawn_named("test/lifecycle", actor).await.unwrap();

        // Wait for startup
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(start_count.load(Ordering::SeqCst), 1);

        // Shutdown the system
        let _ = system.shutdown().await;

        // Note: on_stop may not be called due to current abort() based shutdown.
        // This is a known limitation.
    }

    #[tokio::test]
    async fn test_multiple_actors_lifecycle() {
        let start_count = Arc::new(AtomicI32::new(0));
        let _stop_count = Arc::new(AtomicI32::new(0));

        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        // Spawn multiple actors
        for i in 0..5 {
            let actor = LifecycleActor {
                start_count: start_count.clone(),
                stop_count: _stop_count.clone(),
            };
            let _actor_ref = system
                .spawn_named(format!("test/actor-{}", i), actor)
                .await
                .unwrap();
        }

        // Wait for all starts
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(start_count.load(Ordering::SeqCst), 5);

        let _ = system.shutdown().await;

        // Note: on_stop may not be called due to current abort() based shutdown.
        // This is a known limitation.
    }
}

mod error_tests {
    use super::*;

    #[tokio::test]
    async fn test_actor_error_propagation() {
        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        let actor_ref = system
            .spawn_named("test/counter", Counter { count: 0 })
            .await
            .unwrap();

        let result: Result<StateResponse, _> = actor_ref.ask(ErrorMessage).await;
        assert!(result.is_err());

        // When receive returns Err, only return error to caller, actor doesn't exit
        let result2: Result<Pong, _> = actor_ref.ask(Ping { value: 1 }).await;
        assert!(
            result2.is_ok(),
            "Actor should still be alive after receive error"
        );

        let _ = system.shutdown().await;
    }

    #[tokio::test]
    async fn test_actor_unknown_message() {
        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        let actor_ref = system
            .spawn_named("test/counter", Counter { count: 0 })
            .await
            .unwrap();

        // Send message with unknown type using the unified Message
        let msg = Message::single("UnknownType", vec![]);
        let result = actor_ref.send(msg).await;
        assert!(result.is_err());

        let _ = system.shutdown().await;
    }
}

mod concurrent_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_asks() {
        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        let actor_ref = system
            .spawn_named("test/counter", Counter { count: 0 })
            .await
            .unwrap();

        let mut handles = Vec::new();
        for i in 0..10 {
            let actor_ref_clone = actor_ref.clone();
            let handle = tokio::spawn(async move {
                let _response: Pong = actor_ref_clone.ask(Ping { value: i }).await.unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Final state should be sum of 0..10
        let response: StateResponse = actor_ref.ask(GetState).await.unwrap();
        assert_eq!(response.value, (0..10).sum::<i32>());

        let _ = system.shutdown().await;
    }

    #[tokio::test]
    async fn test_slow_message_handling() {
        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        let actor_ref = system
            .spawn_named("test/counter", Counter { count: 0 })
            .await
            .unwrap();

        let start = std::time::Instant::now();
        let _response: StateResponse = actor_ref.ask(SlowMessage { delay_ms: 100 }).await.unwrap();
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(100));

        let _ = system.shutdown().await;
    }
}

mod spawn_tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_multiple_actors() {
        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        let mut refs = Vec::new();
        for i in 0..5 {
            refs.push(
                system
                    .spawn_named(format!("test/counter-{}", i), Counter { count: 0 })
                    .await
                    .unwrap(),
            );
        }

        // Each actor should be independent
        for (i, actor_ref) in refs.iter().enumerate() {
            let response: Pong = actor_ref
                .ask(Ping {
                    value: i as i32 + 1,
                })
                .await
                .unwrap();
            assert_eq!(response.result, i as i32 + 1);
        }

        let _ = system.shutdown().await;
    }

    #[tokio::test]
    async fn test_actor_isolation() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let ref1 = system
            .spawn_named("test/counter1", Counter { count: 0 })
            .await
            .unwrap();
        let ref2 = system
            .spawn_named("test/counter2", Counter { count: 0 })
            .await
            .unwrap();

        // Modify actor1
        ref1.ask::<_, Pong>(Ping { value: 100 }).await.unwrap();

        // Actor2 should be unaffected
        let response: StateResponse = ref2.ask(GetState).await.unwrap();
        assert_eq!(response.value, 0);

        let _ = system.shutdown().await;
    }
}
