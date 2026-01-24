use pulsing_actor::actor::ActorId;
use pulsing_actor::prelude::*;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Ping {
    value: i32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Pong {
    value: i32,
}

struct Target;

#[async_trait]
impl Actor for Target {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let ping: Ping = msg.unpack()?;
        Message::pack(&Pong {
            value: ping.value + 1,
        })
    }
}

struct Forwarder {
    target: ActorId,
}

#[async_trait]
impl Actor for Forwarder {
    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let ping: Ping = msg.unpack()?;
        let target_ref = ctx.actor_ref(&self.target).await?;
        let pong: Pong = target_ref.ask(Ping { value: ping.value }).await?;
        Message::pack(&pong)
    }
}

#[tokio::test]
async fn actor_context_can_resolve_actor_ref() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let target_ref = system.spawn_named("test/target", Target).await.unwrap();
    let forwarder_ref = system
        .spawn_named(
            "test/forwarder",
            Forwarder {
                target: *target_ref.id(),
            },
        )
        .await
        .unwrap();

    let resp: Pong = forwarder_ref.ask(Ping { value: 41 }).await.unwrap();
    assert_eq!(resp.value, 42);
}

#[tokio::test]
async fn shutdown_clears_all_actors() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    // Spawn some actors
    let _a1 = system.spawn_named("test/actor1", Target).await.unwrap();
    let _a2 = system.spawn_named("test/actor2", Target).await.unwrap();

    // Verify actors exist
    let names = system.local_actor_names();
    assert!(names.contains(&"test/actor1".to_string()));
    assert!(names.contains(&"test/actor2".to_string()));

    // Shutdown
    system.shutdown().await.unwrap();

    // Verify all actors are cleared
    let names_after = system.local_actor_names();
    assert!(
        names_after.is_empty(),
        "Expected no actors after shutdown, but found: {:?}",
        names_after
    );
}
