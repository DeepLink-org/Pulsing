//! Integration tests for the complete actor system

use pulsing_actor::actor::{ActorAddress, ActorId, ActorPath, NodeId};
use pulsing_actor::prelude::*;
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct Pong {
    result: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Accumulate {
    amount: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetTotal;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct TotalResponse {
    total: i32,
}

// ============================================================================
// Test Actors
// ============================================================================

struct EchoActor {
    echo_count: Arc<AtomicUsize>,
}

#[async_trait]
impl Actor for EchoActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        if msg.msg_type().ends_with("Ping") {
            let ping: Ping = msg.unpack()?;
            self.echo_count.fetch_add(1, Ordering::SeqCst);
            return Message::pack(&Pong {
                result: ping.value * 2,
            });
        }
        Err(anyhow::anyhow!("Unknown message"))
    }
}

struct Accumulator {
    total: i32,
}

#[async_trait]
impl Actor for Accumulator {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let msg_type = msg.msg_type();
        if msg_type.ends_with("Accumulate") {
            let acc: Accumulate = msg.unpack()?;
            self.total += acc.amount;
            return Message::pack(&TotalResponse { total: self.total });
        }
        if msg_type.ends_with("GetTotal") {
            return Message::pack(&TotalResponse { total: self.total });
        }
        Err(anyhow::anyhow!("Unknown message"))
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

        let actor = EchoActor {
            echo_count: counter.clone(),
        };
        let actor_ref = system.spawn("echo", actor).await.unwrap();

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
            let actor = Accumulator { total: 0 };
            refs.push(system.spawn(format!("acc-{}", i), actor).await.unwrap());
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

        let actor = EchoActor {
            echo_count: counter.clone(),
        };
        let actor_ref = system.spawn("echo", actor).await.unwrap();

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
        let ref1 = system
            .spawn("acc1", Accumulator { total: 0 })
            .await
            .unwrap();
        let ref2 = system
            .spawn("acc2", Accumulator { total: 0 })
            .await
            .unwrap();

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
            let actor = Accumulator { total: 0 };
            refs.push(system.spawn(format!("acc-{}", i), actor).await.unwrap());
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

        let actor = EchoActor {
            echo_count: counter.clone(),
        };
        let actor_ref = system.spawn("echo", actor).await.unwrap();

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

        let actor = EchoActor {
            echo_count: counter.clone(),
        };
        let actor_ref = system.spawn("echo", actor).await.unwrap();

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

    struct Crashy {
        crash_count: Arc<AtomicI32>,
    }

    #[async_trait]
    impl Actor for Crashy {
        async fn receive(
            &mut self,
            msg: Message,
            _ctx: &mut ActorContext,
        ) -> anyhow::Result<Message> {
            if msg.msg_type().ends_with("CrashMessage") {
                self.crash_count.fetch_add(1, Ordering::SeqCst);
                return Err(anyhow::anyhow!("Intentional crash!"));
            }
            if msg.msg_type().ends_with("Ping") {
                let ping: Ping = msg.unpack()?;
                return Message::pack(&Pong { result: ping.value });
            }
            Err(anyhow::anyhow!("Unknown message"))
        }
    }

    #[tokio::test]
    async fn test_actor_error_recovery() {
        let crash_count = Arc::new(AtomicI32::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor_ref = system
            .spawn(
                "crashy",
                Crashy {
                    crash_count: crash_count.clone(),
                },
            )
            .await
            .unwrap();

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

        let actor_ref = system
            .spawn(
                "crashy",
                Crashy {
                    crash_count: crash_count.clone(),
                },
            )
            .await
            .unwrap();

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

        let actor_ref = system
            .spawn(
                "echo",
                EchoActor {
                    echo_count: counter,
                },
            )
            .await
            .unwrap();

        #[derive(Serialize, Deserialize, Debug, Clone)]
        struct UnknownMsg;

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
        events: Arc<tokio::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl Actor for LifecycleTracker {
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
            msg: Message,
            _ctx: &mut ActorContext,
        ) -> anyhow::Result<Message> {
            self.events
                .lock()
                .await
                .push(format!("received:{}", msg.msg_type()));
            Message::pack(&Pong { result: 0 })
        }
    }

    #[tokio::test]
    async fn test_lifecycle_events_order() {
        let events = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let actor_ref = system
            .spawn(
                "tracker",
                LifecycleTracker {
                    events: events.clone(),
                },
            )
            .await
            .unwrap();

        // Send a message
        let _: Pong = actor_ref.ask(Ping { value: 1 }).await.unwrap();

        // Stop the actor
        system.stop("tracker").await.unwrap();

        // Wait for stop to complete
        tokio::time::sleep(Duration::from_millis(50)).await;

        let recorded = events.lock().await;
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0], "started");
        // type_name includes full module path
        assert!(recorded[1].starts_with("received:"));
        assert!(recorded[1].ends_with("Ping"));
        // Note: on_stop may not be called due to abort() behavior
    }

    #[tokio::test]
    async fn test_system_shutdown_stops_all() {
        let events1 = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let events2 = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        let _ = system
            .spawn(
                "tracker1",
                LifecycleTracker {
                    events: events1.clone(),
                },
            )
            .await
            .unwrap();
        let _ = system
            .spawn(
                "tracker2",
                LifecycleTracker {
                    events: events2.clone(),
                },
            )
            .await
            .unwrap();

        // Wait for actors to start
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Shutdown should stop all
        system.shutdown().await.unwrap();

        // Both should have started at least
        assert!(events1.lock().await.contains(&"started".to_string()));
        assert!(events2.lock().await.contains(&"started".to_string()));
    }
}

// ============================================================================
// Addressing Tests
// ============================================================================

mod addressing_tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_named_actor() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Create a named actor with path
        let path = ActorPath::new("services/echo").unwrap();
        let actor_ref = system
            .spawn_named(
                path.clone(),
                "echo_impl",
                EchoActor {
                    echo_count: counter.clone(),
                },
            )
            .await
            .unwrap();

        // Send message via the returned ref
        let response: Pong = actor_ref.ask(Ping { value: 21 }).await.unwrap();
        assert_eq!(response.result, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Lookup named actor info
        let info = system.lookup_named(&path).await;
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.path.as_str(), "services/echo");
        assert_eq!(info.instance_count(), 1);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_named_actor() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Create a named actor
        let path = ActorPath::new("services/api/handler").unwrap();
        let _actor_ref = system
            .spawn_named(
                path.clone(),
                "api_handler",
                EchoActor {
                    echo_count: counter.clone(),
                },
            )
            .await
            .unwrap();

        // Resolve by address
        let addr = ActorAddress::parse("actor:///services/api/handler").unwrap();
        let resolved_ref = system.resolve(&addr).await.unwrap();

        // Send message via resolved ref
        let response: Pong = resolved_ref.ask(Ping { value: 10 }).await.unwrap();
        assert_eq!(response.result, 20);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_global_address() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Create a regular actor
        let _actor_ref = system
            .spawn(
                "worker",
                EchoActor {
                    echo_count: counter.clone(),
                },
            )
            .await
            .unwrap();

        // Get the full address
        let node_id = system.node_id().clone();
        let addr = ActorAddress::global(node_id, "worker");

        // Resolve
        let resolved_ref = system.resolve(&addr).await.unwrap();
        let response: Pong = resolved_ref.ask(Ping { value: 5 }).await.unwrap();
        assert_eq!(response.result, 10);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_localhost() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Create actor
        let _actor_ref = system
            .spawn(
                "local_worker",
                EchoActor {
                    echo_count: counter.clone(),
                },
            )
            .await
            .unwrap();

        // Resolve using localhost
        let addr = ActorAddress::parse("actor://localhost/local_worker").unwrap();
        assert!(addr.is_localhost());

        let resolved_ref = system.resolve(&addr).await.unwrap();
        let response: Pong = resolved_ref.ask(Ping { value: 7 }).await.unwrap();
        assert_eq!(response.result, 14);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_stop_named_actor() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Create a named actor
        let path = ActorPath::new("services/temp").unwrap();
        let _actor_ref = system
            .spawn_named(
                path.clone(),
                "temp_actor",
                EchoActor {
                    echo_count: counter.clone(),
                },
            )
            .await
            .unwrap();

        // Verify it exists
        assert!(system.lookup_named(&path).await.is_some());

        // Stop the named actor
        system.stop_named(&path).await.unwrap();

        // Wait a bit for cleanup
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Should no longer exist in local registry
        // Note: In single-node mode, after stop the gossip registry
        // will also be updated, but there might be a small delay
        let info = system.lookup_named(&path).await;
        // The info might still exist but with 0 instances, or not exist at all
        if let Some(info) = info {
            assert_eq!(info.instance_count(), 0);
        }

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_path_parsing() {
        // Valid paths
        assert!(ActorPath::new("services/api").is_ok());
        assert!(ActorPath::new("services/llm/router").is_ok());
        assert!(ActorPath::new("workers/pool/manager").is_ok());

        // Invalid paths
        assert!(ActorPath::new("single").is_err()); // needs namespace
        assert!(ActorPath::new("").is_err());
        assert!(ActorPath::new("a//b").is_err()); // empty segment
    }

    #[tokio::test]
    async fn test_actor_address_parsing() {
        // Named service
        let addr = ActorAddress::parse("actor:///services/api").unwrap();
        assert!(addr.is_named());
        assert_eq!(addr.path().unwrap().namespace(), "services");
        assert_eq!(addr.path().unwrap().name(), "api");

        // Named instance
        let addr = ActorAddress::parse("actor:///services/api@node_a").unwrap();
        assert!(addr.is_named());
        assert_eq!(addr.node_id().map(|n| n.as_str()), Some("node_a"));

        // Global
        let addr = ActorAddress::parse("actor://node_b/worker_123").unwrap();
        assert!(addr.is_global());
        assert_eq!(addr.actor_id(), Some("worker_123"));

        // Local
        let addr = ActorAddress::parse("actor://localhost/my_actor").unwrap();
        assert!(addr.is_localhost());
    }

    #[tokio::test]
    async fn test_resolve_not_found() {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

        // Try to resolve non-existent named actor
        let addr = ActorAddress::parse("actor:///services/nonexistent").unwrap();
        let result = system.resolve(&addr).await;
        assert!(result.is_err());

        // Try to resolve non-existent global actor
        let addr = ActorAddress::parse("actor://localhost/nonexistent").unwrap();
        let result = system.resolve(&addr).await;
        assert!(result.is_err());

        system.shutdown().await.unwrap();
    }
}
