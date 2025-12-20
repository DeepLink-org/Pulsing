//! Actor core functionality tests

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

impl Message for Ping {
    fn type_id() -> &'static str {
        "Ping"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct Pong {
    result: i32,
}

impl Message for Pong {
    fn type_id() -> &'static str {
        "Pong"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Increment {
    amount: i32,
}

impl Message for Increment {
    fn type_id() -> &'static str {
        "Increment"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetState;

impl Message for GetState {
    fn type_id() -> &'static str {
        "GetState"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct StateResponse {
    value: i32,
}

impl Message for StateResponse {
    fn type_id() -> &'static str {
        "StateResponse"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SlowMessage {
    delay_ms: u64,
}

impl Message for SlowMessage {
    fn type_id() -> &'static str {
        "SlowMessage"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ErrorMessage;

impl Message for ErrorMessage {
    fn type_id() -> &'static str {
        "ErrorMessage"
    }
}

// ============================================================================
// Test Actors
// ============================================================================

/// Simple counter actor for basic tests
struct CounterActor {
    id: ActorId,
    count: i32,
    started: bool,
    stopped: bool,
}

impl CounterActor {
    fn new(name: &str) -> Self {
        Self {
            id: ActorId::local(name),
            count: 0,
            started: false,
            stopped: false,
        }
    }
}

#[async_trait]
impl Actor for CounterActor {
    fn id(&self) -> &ActorId {
        &self.id
    }

    async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        self.started = true;
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        self.stopped = true;
        Ok(())
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "Ping" => {
                let ping: Ping = msg.into_message()?;
                self.count += ping.value;
                RawMessage::from_message(&Pong { result: self.count })
            }
            "Increment" => {
                let inc: Increment = msg.into_message()?;
                self.count += inc.amount;
                RawMessage::from_message(&StateResponse { value: self.count })
            }
            "GetState" => RawMessage::from_message(&StateResponse { value: self.count }),
            "SlowMessage" => {
                let slow: SlowMessage = msg.into_message()?;
                tokio::time::sleep(Duration::from_millis(slow.delay_ms)).await;
                RawMessage::from_message(&StateResponse { value: self.count })
            }
            "ErrorMessage" => Err(anyhow::anyhow!("Intentional error for testing")),
            _ => Err(anyhow::anyhow!("Unknown message type: {}", msg.msg_type)),
        }
    }
}

/// Actor that tracks lifecycle events with shared state
struct LifecycleActor {
    id: ActorId,
    start_count: Arc<AtomicI32>,
    stop_count: Arc<AtomicI32>,
}

impl LifecycleActor {
    fn new(name: &str, start_count: Arc<AtomicI32>, stop_count: Arc<AtomicI32>) -> Self {
        Self {
            id: ActorId::local(name),
            start_count,
            stop_count,
        }
    }
}

#[async_trait]
impl Actor for LifecycleActor {
    fn id(&self) -> &ActorId {
        &self.id
    }

    async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        self.start_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        self.stop_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "Ping" => RawMessage::from_message(&Pong { result: 0 }),
            _ => Err(anyhow::anyhow!("Unknown message")),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

mod actor_spawn_tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_single_actor() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter");
        let actor_ref = system.spawn(actor).await.unwrap();

        assert!(actor_ref.is_local());
        assert_eq!(actor_ref.id().name, "counter");

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_spawn_multiple_actors() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let mut refs = Vec::new();
        for i in 0..10 {
            let actor = CounterActor::new(&format!("counter-{}", i));
            let actor_ref = system.spawn(actor).await.unwrap();
            refs.push(actor_ref);
        }

        assert_eq!(refs.len(), 10);
        assert_eq!(system.local_actor_names().len(), 10);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_lifecycle_callbacks() {
        let start_count = Arc::new(AtomicI32::new(0));
        let stop_count = Arc::new(AtomicI32::new(0));

        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = LifecycleActor::new("lifecycle", start_count.clone(), stop_count.clone());
        let _actor_ref = system.spawn(actor).await.unwrap();

        // on_start should have been called
        assert_eq!(start_count.load(Ordering::SeqCst), 1);
        assert_eq!(stop_count.load(Ordering::SeqCst), 0);

        // Stop the actor
        system.stop("lifecycle").await.unwrap();

        // Give time for on_stop to be called
        tokio::time::sleep(Duration::from_millis(50)).await;

        system.shutdown().await.unwrap();
    }
}

mod actor_messaging_tests {
    use super::*;

    #[tokio::test]
    async fn test_ask_single_message() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter");
        let actor_ref = system.spawn(actor).await.unwrap();

        let response: Pong = actor_ref.ask(Ping { value: 10 }).await.unwrap();
        assert_eq!(response.result, 10);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_ask_multiple_messages_sequential() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter");
        let actor_ref = system.spawn(actor).await.unwrap();

        for i in 1..=10 {
            let response: Pong = actor_ref.ask(Ping { value: i }).await.unwrap();
            assert_eq!(response.result, (1..=i).sum::<i32>());
        }

        // Final count should be 1+2+3+...+10 = 55
        let state: StateResponse = actor_ref.ask(GetState).await.unwrap();
        assert_eq!(state.value, 55);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_tell_fire_and_forget() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter");
        let actor_ref = system.spawn(actor).await.unwrap();

        // Send multiple tells
        for i in 1..=5 {
            actor_ref.tell(Increment { amount: i }).await.unwrap();
        }

        // Wait for messages to be processed
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify state
        let state: StateResponse = actor_ref.ask(GetState).await.unwrap();
        assert_eq!(state.value, 15); // 1+2+3+4+5

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_ask_concurrent_messages() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter");
        let actor_ref = system.spawn(actor).await.unwrap();

        // Send concurrent requests
        let mut handles = Vec::new();
        for i in 1..=10 {
            let ref_clone = actor_ref.clone();
            let handle = tokio::spawn(async move {
                let _: Pong = ref_clone.ask(Ping { value: i }).await.unwrap();
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Final state should be sum of 1..=10
        let state: StateResponse = actor_ref.ask(GetState).await.unwrap();
        assert_eq!(state.value, 55);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_error_handling() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter");
        let actor_ref = system.spawn(actor).await.unwrap();

        // Send error message
        let result: anyhow::Result<Pong> = actor_ref.ask(ErrorMessage).await;
        assert!(result.is_err());

        // Actor should still be alive and functioning
        let response: Pong = actor_ref.ask(Ping { value: 1 }).await.unwrap();
        assert_eq!(response.result, 1);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_slow_message_handling() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter");
        let actor_ref = system.spawn(actor).await.unwrap();

        let start = std::time::Instant::now();
        let _: StateResponse = actor_ref
            .ask(SlowMessage { delay_ms: 100 })
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(100));

        system.shutdown().await.unwrap();
    }
}

mod actor_ref_tests {
    use super::*;

    #[tokio::test]
    async fn test_actor_ref_clone() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("counter");
        let actor_ref = system.spawn(actor).await.unwrap();

        // Clone the ref and use both
        let ref1 = actor_ref.clone();
        let ref2 = actor_ref.clone();

        let _: Pong = ref1.ask(Ping { value: 10 }).await.unwrap();
        let _: Pong = ref2.ask(Ping { value: 20 }).await.unwrap();

        let state: StateResponse = actor_ref.ask(GetState).await.unwrap();
        assert_eq!(state.value, 30);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_get_actor_ref_by_id() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CounterActor::new("my-counter");
        let original_ref = system.spawn(actor).await.unwrap();

        // Get reference by ID
        let retrieved_ref = system.actor_ref(original_ref.id()).await.unwrap();

        // Both refs should work
        let _: Pong = original_ref.ask(Ping { value: 5 }).await.unwrap();
        let _: Pong = retrieved_ref.ask(Ping { value: 10 }).await.unwrap();

        let state: StateResponse = original_ref.ask(GetState).await.unwrap();
        assert_eq!(state.value, 15);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_not_found() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let fake_id = ActorId::new(system.node_id().clone(), "nonexistent");
        let result = system.actor_ref(&fake_id).await;

        assert!(result.is_err());

        system.shutdown().await.unwrap();
    }
}

mod system_tests {
    use super::*;

    #[tokio::test]
    async fn test_system_shutdown() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Spawn some actors
        for i in 0..5 {
            let actor = CounterActor::new(&format!("counter-{}", i));
            let _ = system.spawn(actor).await.unwrap();
        }

        assert_eq!(system.local_actor_names().len(), 5);

        // Shutdown
        system.shutdown().await.unwrap();

        // Local actors should be cleared
        assert_eq!(system.local_actor_names().len(), 0);
    }

    #[tokio::test]
    async fn test_stop_individual_actor() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor1 = CounterActor::new("counter-1");
        let actor2 = CounterActor::new("counter-2");

        let ref1 = system.spawn(actor1).await.unwrap();
        let _ref2 = system.spawn(actor2).await.unwrap();

        assert_eq!(system.local_actor_names().len(), 2);

        // Stop only counter-1
        system.stop("counter-1").await.unwrap();

        assert_eq!(system.local_actor_names().len(), 1);
        assert!(!system.local_actor_names().contains(&"counter-1".to_string()));
        assert!(system.local_actor_names().contains(&"counter-2".to_string()));

        // Sending to stopped actor should fail
        let result: anyhow::Result<Pong> = ref1.ask(Ping { value: 1 }).await;
        assert!(result.is_err());

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_system_cancellation_token() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
        let token = system.cancel_token();

        assert!(!token.is_cancelled());

        system.shutdown().await.unwrap();

        assert!(token.is_cancelled());
    }
}

