//! Tests for spawning behavior-based actors
//!
//! With `IntoActor`, `Behavior<M>` can be directly passed to `spawn()` and `spawn_named()`.
//! The actual `BehaviorWrapper` that bridges Behavior to Actor trait is in `core.rs`.

#[cfg(test)]
mod tests {
    use crate::behavior::{stateful, BehaviorAction};
    use crate::system::{ActorSystem, ActorSystemCoreExt, SystemConfig};
    use serde::Serialize;
    use std::sync::Arc;

    #[derive(Debug, Clone, Serialize, serde::Deserialize)]
    enum TestMsg {
        Ping,
        Increment(i32),
    }

    #[tokio::test]
    async fn test_spawn_behavior_with_into_actor() {
        let system = Arc::new(ActorSystem::new(SystemConfig::standalone()).await.unwrap());

        let counter = stateful(0i32, |count, msg, _ctx| match msg {
            TestMsg::Ping => BehaviorAction::Same,
            TestMsg::Increment(n) => {
                *count += n;
                BehaviorAction::Same
            }
        });

        // Use spawn_named with Behavior directly (via IntoActor)
        let actor_ref = system.spawn_named("test/counter", counter).await.unwrap();

        // Send messages via ActorRef
        actor_ref.tell(TestMsg::Increment(5)).await.unwrap();
        actor_ref.tell(TestMsg::Ping).await.unwrap();

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_spawn_anonymous_behavior() {
        let system = Arc::new(ActorSystem::new(SystemConfig::standalone()).await.unwrap());

        let counter = stateful(0i32, |count, msg, _ctx| match msg {
            TestMsg::Ping => BehaviorAction::Same,
            TestMsg::Increment(n) => {
                *count += n;
                BehaviorAction::Same
            }
        });

        // Use spawn with Behavior directly (via IntoActor)
        let actor_ref = system.spawn(counter).await.unwrap();

        // Send messages via ActorRef
        actor_ref.tell(TestMsg::Increment(10)).await.unwrap();

        system.shutdown().await.unwrap();
    }
}
