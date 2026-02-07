//! Unit tests for TypedRef: tell, ask, ask_timeout, as_untyped, is_alive.

use pulsing_actor::behavior::{stateless, BehaviorAction, BehaviorWrapper};
use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize, Deserialize, Debug)]
struct Ping {
    value: i32,
}

#[derive(Serialize, Deserialize, Debug)]
struct Pong {
    result: i32,
}

#[derive(Serialize, Deserialize, Debug)]
enum RefTestMsg {
    Ping,
}

/// Echo actor that doubles Ping.value for Pong.result
struct EchoActor;

#[async_trait]
impl Actor for EchoActor {
    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        if msg.msg_type().ends_with("Ping") {
            let ping: Ping = msg.unpack()?;
            return Message::pack(&Pong {
                result: ping.value * 2,
            });
        }
        Err(pulsing_actor::error::PulsingError::from(
            pulsing_actor::error::RuntimeError::Other("unknown".to_string()),
        ))
    }
}

#[tokio::test]
async fn test_typed_ref_tell_and_ask() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let echo = system
        .spawn_named("test/typed-echo", EchoActor)
        .await
        .unwrap();
    let typed = pulsing_actor::behavior::TypedRef::<Ping>::new("test/typed-echo", echo.clone());
    assert_eq!(typed.name(), "test/typed-echo");
    assert!(typed.is_alive());

    typed.tell(Ping { value: 1 }).await.unwrap();
    let pong: Pong = typed.ask(Ping { value: 2 }).await.unwrap();
    assert_eq!(pong.result, 4);

    let untyped = typed.as_untyped().unwrap();
    assert!(untyped.id() == echo.id());
    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_typed_ref_ask_timeout_ok() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let echo = system
        .spawn_named("test/typed-echo-timeout", EchoActor)
        .await
        .unwrap();
    let typed = pulsing_actor::behavior::TypedRef::<Ping>::new("test/typed-echo-timeout", echo);
    let pong: Pong = typed
        .ask_timeout(Ping { value: 10 }, Duration::from_secs(2))
        .await
        .unwrap();
    assert_eq!(pong.result, 20);
    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_typed_ref_from_context_typed_ref() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let behavior = stateless(|_msg: RefTestMsg, ctx| {
        Box::pin(async move {
            let other: pulsing_actor::behavior::TypedRef<Ping> =
                ctx.typed_ref("test/typed-echo-from-ctx");
            assert_eq!(other.name(), "test/typed-echo-from-ctx");
            BehaviorAction::Same
        })
    });
    let wrapper: BehaviorWrapper<RefTestMsg> = behavior.into_actor();
    let _echo_ref = system
        .spawn_named("test/typed-echo-from-ctx", EchoActor)
        .await
        .unwrap();
    let actor = system.spawn_named("test/caller", wrapper).await.unwrap();
    let _: () = actor.ask(RefTestMsg::Ping).await.unwrap();
    system.shutdown().await.unwrap();
}
