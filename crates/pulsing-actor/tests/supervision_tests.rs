use pulsing_actor::prelude::*;
use pulsing_actor::supervision::{BackoffStrategy, SupervisionSpec};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

struct FailingActor {
    counter: Arc<AtomicU32>,
    fail_at: u32,
}

#[async_trait]
impl Actor for FailingActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst) + 1;

        if count == self.fail_at {
            return Err(anyhow::anyhow!("Boom!"));
        }

        // Echo
        Ok(msg)
    }
}

#[tokio::test]
async fn test_restart_on_failure() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let counter = Arc::new(AtomicU32::new(0));

    let counter_clone = counter.clone();
    let factory = move || {
        Ok(FailingActor {
            counter: counter_clone.clone(),
            fail_at: 2, // Fail on 2nd message
        })
    };

    let spec = SupervisionSpec::on_failure()
        .with_max_restarts(3)
        .with_backoff(BackoffStrategy::exponential(
            Duration::from_millis(10),
            Duration::from_millis(100),
        ));

    let options = SpawnOptions::new().supervision(spec);

    let actor_ref = system
        .spawn_named_factory("test/failing", factory, options)
        .await
        .unwrap();

    // 1st message - success
    let resp = actor_ref.send(Message::single("ping", b"1")).await;
    assert!(resp.is_ok());

    // 2nd message - failure (should crash and restart)
    let resp = actor_ref.send(Message::single("ping", b"2")).await;
    assert!(resp.is_err()); // The ask fails because the actor crashed handling it

    // Wait a bit for restart
    tokio::time::sleep(Duration::from_millis(50)).await;

    // 3rd message - success (new instance)
    // Note: counter is shared, so it will continue from 2 -> 3
    let resp = actor_ref.send(Message::single("ping", b"3")).await;
    assert!(resp.is_ok());

    let msg = resp.unwrap();
    if let Message::Single { data, .. } = msg {
        assert_eq!(data, b"3");
    } else {
        panic!("expected single message");
    }

    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn test_max_restarts_exceeded() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let counter = Arc::new(AtomicU32::new(0));

    let counter_clone = counter.clone();
    // Fail immediately
    let factory = move || {
        counter_clone.fetch_add(1, Ordering::SeqCst);
        Ok(FailingActor {
            counter: Arc::new(AtomicU32::new(0)), // Unused
            fail_at: 1,                           // Fail immediately
        })
    };

    let spec = SupervisionSpec::on_failure()
        .with_max_restarts(2) // Allow 2 restarts
        .with_backoff(BackoffStrategy {
            min: Duration::from_millis(1),
            max: Duration::from_millis(1),
            jitter: 0.0,
            factor: 1.0,
        });

    let options = SpawnOptions::new().supervision(spec);

    let actor_ref = system
        .spawn_named_factory("test/crashing", factory, options)
        .await
        .unwrap();

    // 1st crash
    let _ = actor_ref.send(Message::single("ping", b"1")).await;
    tokio::time::sleep(Duration::from_millis(10)).await;

    // 2nd crash
    let _ = actor_ref.send(Message::single("ping", b"2")).await;
    tokio::time::sleep(Duration::from_millis(10)).await;

    // 3rd crash
    let _ = actor_ref.send(Message::single("ping", b"3")).await;
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Should be dead now (Initial start + 2 restarts = 3 failures. Next attempt stops.)
    // Wait for supervision loop to exit
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Send message to dead actor
    let resp = actor_ref.send(Message::single("ping", b"4")).await;
    assert!(resp.is_err()); // Mailbox closed

    // Check factory calls: Initial + 2 restarts = 3 calls
    // Actually, if it crashes 3 times:
    // 1. Start (count=1), Receive -> Crash
    // 2. Restart 1 (count=2), Receive -> Crash
    // 3. Restart 2 (count=3), Receive -> Crash
    // 4. Max restarts exceeded -> Stop
    // So factory called 3 times.
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}
