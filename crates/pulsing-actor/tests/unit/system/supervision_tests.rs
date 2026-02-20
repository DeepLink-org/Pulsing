use pulsing_actor::error::{PulsingError, RuntimeError};
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
    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst) + 1;

        if count == self.fail_at {
            return Err(PulsingError::from(RuntimeError::Other("Boom!".into())));
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

    let actor_ref = system
        .spawning()
        .name("test/failing")
        .supervision(spec)
        .spawn_factory(factory)
        .await
        .unwrap();

    // 1st message - success
    let resp = actor_ref.send(Message::single("ping", b"1")).await;
    assert!(resp.is_ok());

    // 2nd message - receive returns Err, error returned to caller, actor doesn't exit or restart
    let resp = actor_ref.send(Message::single("ping", b"2")).await;
    assert!(resp.is_err());

    // 3rd message - same instance still alive, continues processing
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
    // receive returning Err doesn't cause actor to exit, so no restart is triggered; factory only called once
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let counter = Arc::new(AtomicU32::new(0));

    let counter_clone = counter.clone();
    let factory = move || {
        counter_clone.fetch_add(1, Ordering::SeqCst);
        Ok(FailingActor {
            counter: Arc::new(AtomicU32::new(0)),
            fail_at: 1, // 1st message returns Err
        })
    };

    let spec = SupervisionSpec::on_failure()
        .with_max_restarts(2)
        .with_backoff(BackoffStrategy {
            min: Duration::from_millis(1),
            max: Duration::from_millis(1),
            jitter: 0.0,
            factor: 1.0,
        });

    let actor_ref = system
        .spawning()
        .name("test/crashing")
        .supervision(spec)
        .spawn_factory(factory)
        .await
        .unwrap();

    // 1st message: receive returns Err, only return error to caller, actor doesn't exit
    let r1 = actor_ref.send(Message::single("ping", b"1")).await;
    assert!(r1.is_err());
    assert_eq!(counter.load(Ordering::SeqCst), 1); // factory only called once

    // 2nd message: same instance, count=2 != fail_at(1), returns Ok
    let r2 = actor_ref.send(Message::single("ping", b"2")).await;
    assert!(r2.is_ok());
    assert_eq!(counter.load(Ordering::SeqCst), 1); // no restart
}
