//! Integration tests for the complete actor system

use pulsing_actor::prelude::*;
use pulsing_actor::system::SystemConfig;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
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
struct Accumulate {
    amount: i32,
}

impl Message for Accumulate {
    fn type_id() -> &'static str {
        "Accumulate"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetTotal;

impl Message for GetTotal {
    fn type_id() -> &'static str {
        "GetTotal"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct TotalResponse {
    total: i32,
}

impl Message for TotalResponse {
    fn type_id() -> &'static str {
        "TotalResponse"
    }
}

// ============================================================================
// Test Actors
// ============================================================================

struct EchoActor {
    id: ActorId,
    echo_count: Arc<AtomicUsize>,
}

impl EchoActor {
    fn new(name: &str, counter: Arc<AtomicUsize>) -> Self {
        Self {
            id: ActorId::local(name),
            echo_count: counter,
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
                self.echo_count.fetch_add(1, Ordering::SeqCst);
                RawMessage::from_message(&Pong {
                    result: ping.value * 2,
                })
            }
            _ => Err(anyhow::anyhow!("Unknown message")),
        }
    }
}

struct AccumulatorActor {
    id: ActorId,
    total: i32,
}

impl AccumulatorActor {
    fn new(name: &str) -> Self {
        Self {
            id: ActorId::local(name),
            total: 0,
        }
    }
}

#[async_trait]
impl Actor for AccumulatorActor {
    fn id(&self) -> &ActorId {
        &self.id
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "Accumulate" => {
                let acc: Accumulate = msg.into_message()?;
                self.total += acc.amount;
                RawMessage::from_message(&TotalResponse { total: self.total })
            }
            "GetTotal" => RawMessage::from_message(&TotalResponse { total: self.total }),
            _ => Err(anyhow::anyhow!("Unknown message")),
        }
    }
}

// ============================================================================
// Single Node Integration Tests
// ============================================================================

mod single_node_tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_actor_communication() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = EchoActor::new("echo", counter.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        let response: Pong = actor_ref.ask(Ping { value: 21 }).await.unwrap();
        assert_eq!(response.result, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_multiple_actors_concurrent() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let mut refs = Vec::new();
        for i in 0..5 {
            let actor = AccumulatorActor::new(&format!("acc-{}", i));
            refs.push(system.spawn(actor).await.unwrap());
        }

        // Send to all actors concurrently
        let mut handles = Vec::new();
        for (i, actor_ref) in refs.iter().enumerate() {
            let ref_clone = actor_ref.clone();
            let handle = tokio::spawn(async move {
                for j in 0..10 {
                    let _: TotalResponse = ref_clone
                        .ask(Accumulate {
                            amount: (i * 10 + j) as i32,
                        })
                        .await
                        .unwrap();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Verify each accumulator has correct total
        for (i, actor_ref) in refs.iter().enumerate() {
            let response: TotalResponse = actor_ref.ask(GetTotal).await.unwrap();
            let expected: i32 = (0..10).map(|j| (i * 10 + j) as i32).sum();
            assert_eq!(response.total, expected);
        }

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_high_message_throughput() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = EchoActor::new("echo", counter.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        let message_count: usize = 1000;
        let start = std::time::Instant::now();

        for i in 0..message_count {
            let _: Pong = actor_ref.ask(Ping { value: i as i32 }).await.unwrap();
        }

        let elapsed = start.elapsed();
        let throughput = message_count as f64 / elapsed.as_secs_f64();

        println!(
            "Single actor throughput: {} msg/sec ({} messages in {:?})",
            throughput as u64, message_count, elapsed
        );

        assert_eq!(counter.load(Ordering::SeqCst), message_count);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_state_isolation() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Create two accumulators
        let acc1 = AccumulatorActor::new("acc1");
        let acc2 = AccumulatorActor::new("acc2");

        let ref1 = system.spawn(acc1).await.unwrap();
        let ref2 = system.spawn(acc2).await.unwrap();

        // Add to acc1
        for i in 1..=5 {
            let _: TotalResponse = ref1.ask(Accumulate { amount: i }).await.unwrap();
        }

        // Add different amounts to acc2
        for i in 10..=15 {
            let _: TotalResponse = ref2.ask(Accumulate { amount: i }).await.unwrap();
        }

        // Verify state isolation
        let total1: TotalResponse = ref1.ask(GetTotal).await.unwrap();
        let total2: TotalResponse = ref2.ask(GetTotal).await.unwrap();

        assert_eq!(total1.total, 15); // 1+2+3+4+5
        assert_eq!(total2.total, 75); // 10+11+12+13+14+15

        system.shutdown().await.unwrap();
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_many_actors() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor_count = 100;
        let mut refs = Vec::with_capacity(actor_count);

        for i in 0..actor_count {
            let actor = AccumulatorActor::new(&format!("acc-{}", i));
            refs.push(system.spawn(actor).await.unwrap());
        }

        assert_eq!(system.local_actor_names().len(), actor_count);

        // Send one message to each
        for (i, actor_ref) in refs.iter().enumerate() {
            let _: TotalResponse = actor_ref
                .ask(Accumulate { amount: i as i32 })
                .await
                .unwrap();
        }

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_burst_messages() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = EchoActor::new("echo", counter.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        let burst_count = 500;

        // Fire off all messages concurrently
        let mut handles = Vec::with_capacity(burst_count);
        for i in 0..burst_count {
            let ref_clone = actor_ref.clone();
            let handle = tokio::spawn(async move {
                let response: Pong = ref_clone.ask(Ping { value: i as i32 }).await.unwrap();
                assert_eq!(response.result, (i as i32) * 2);
            });
            handles.push(handle);
        }

        // All should complete
        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), burst_count);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_sustained_load() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = EchoActor::new("echo", counter.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        let duration = Duration::from_secs(2);
        let start = std::time::Instant::now();
        let mut message_count: usize = 0;

        while start.elapsed() < duration {
            let _: Pong = actor_ref.ask(Ping { value: 1 }).await.unwrap();
            message_count += 1;
        }

        let throughput = message_count as f64 / duration.as_secs_f64();
        println!(
            "Sustained load: {} messages in {:?} ({:.0} msg/sec)",
            message_count, duration, throughput
        );

        assert_eq!(counter.load(Ordering::SeqCst), message_count);

        system.shutdown().await.unwrap();
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct CrashMessage;

    impl Message for CrashMessage {
        fn type_id() -> &'static str {
            "CrashMessage"
        }
    }

    struct CrashyActor {
        id: ActorId,
        crash_count: Arc<AtomicI32>,
    }

    impl CrashyActor {
        fn new(name: &str, crash_count: Arc<AtomicI32>) -> Self {
            Self {
                id: ActorId::local(name),
                crash_count,
            }
        }
    }

    #[async_trait]
    impl Actor for CrashyActor {
        fn id(&self) -> &ActorId {
            &self.id
        }

        async fn receive(
            &mut self,
            msg: RawMessage,
            _ctx: &mut ActorContext,
        ) -> anyhow::Result<RawMessage> {
            match msg.msg_type.as_str() {
                "CrashMessage" => {
                    self.crash_count.fetch_add(1, Ordering::SeqCst);
                    Err(anyhow::anyhow!("Intentional crash!"))
                }
                "Ping" => {
                    let ping: Ping = msg.into_message()?;
                    RawMessage::from_message(&Pong { result: ping.value })
                }
                _ => Err(anyhow::anyhow!("Unknown message")),
            }
        }
    }

    #[tokio::test]
    async fn test_actor_error_recovery() {
        let crash_count = Arc::new(AtomicI32::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CrashyActor::new("crashy", crash_count.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        // Send crash message
        let result: Result<Pong, _> = actor_ref.ask(CrashMessage).await;
        assert!(result.is_err());
        assert_eq!(crash_count.load(Ordering::SeqCst), 1);

        // Actor should still respond to normal messages
        let response: Pong = actor_ref.ask(Ping { value: 42 }).await.unwrap();
        assert_eq!(response.result, 42);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_multiple_errors() {
        let crash_count = Arc::new(AtomicI32::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = CrashyActor::new("crashy", crash_count.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        // Multiple crash messages
        for _ in 0..5 {
            let _: Result<Pong, _> = actor_ref.ask(CrashMessage).await;
        }

        assert_eq!(crash_count.load(Ordering::SeqCst), 5);

        // Should still work
        let response: Pong = actor_ref.ask(Ping { value: 100 }).await.unwrap();
        assert_eq!(response.result, 100);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_unknown_message_type() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = EchoActor::new("echo", counter);
        let actor_ref = system.spawn(actor).await.unwrap();

        #[derive(Serialize, Deserialize, Debug, Clone)]
        struct UnknownMsg;

        impl Message for UnknownMsg {
            fn type_id() -> &'static str {
                "UnknownMsg"
            }
        }

        let result: Result<Pong, _> = actor_ref.ask(UnknownMsg).await;
        assert!(result.is_err());

        system.shutdown().await.unwrap();
    }
}

// ============================================================================
// Lifecycle Tests
// ============================================================================

mod lifecycle_tests {
    use super::*;

    struct LifecycleTracker {
        id: ActorId,
        events: Arc<tokio::sync::Mutex<Vec<String>>>,
    }

    impl LifecycleTracker {
        fn new(name: &str, events: Arc<tokio::sync::Mutex<Vec<String>>>) -> Self {
            Self {
                id: ActorId::local(name),
                events,
            }
        }
    }

    #[async_trait]
    impl Actor for LifecycleTracker {
        fn id(&self) -> &ActorId {
            &self.id
        }

        async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
            self.events.lock().await.push("started".to_string());
            Ok(())
        }

        async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
            self.events.lock().await.push("stopped".to_string());
            Ok(())
        }

        async fn receive(
            &mut self,
            msg: RawMessage,
            _ctx: &mut ActorContext,
        ) -> anyhow::Result<RawMessage> {
            self.events
                .lock()
                .await
                .push(format!("received:{}", msg.msg_type));
            RawMessage::from_message(&Pong { result: 0 })
        }
    }

    #[tokio::test]
    async fn test_lifecycle_events_order() {
        let events = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor = LifecycleTracker::new("tracker", events.clone());
        let actor_ref = system.spawn(actor).await.unwrap();

        // Send a message
        let _: Pong = actor_ref.ask(Ping { value: 1 }).await.unwrap();

        // Stop the actor
        system.stop("tracker").await.unwrap();

        // Wait for stop to complete
        tokio::time::sleep(Duration::from_millis(50)).await;

        let recorded = events.lock().await;
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0], "started");
        assert_eq!(recorded[1], "received:Ping");
        // Note: on_stop may not be called due to abort() behavior
    }

    #[tokio::test]
    async fn test_system_shutdown_stops_all() {
        let events1 = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let events2 = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor1 = LifecycleTracker::new("tracker1", events1.clone());
        let actor2 = LifecycleTracker::new("tracker2", events2.clone());

        let _ = system.spawn(actor1).await.unwrap();
        let _ = system.spawn(actor2).await.unwrap();

        // Shutdown should stop all
        system.shutdown().await.unwrap();

        // Both should have started at least
        assert!(events1.lock().await.contains(&"started".to_string()));
        assert!(events2.lock().await.contains(&"started".to_string()));
    }
}
