//! Unit tests for behavior core: Behavior creation, message handling, BehaviorAction.

use pulsing_actor::behavior::{stateful, stateless, Behavior, BehaviorAction, BehaviorWrapper};
use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
enum TestMsg {
    Ping,
    Add(i32),
    Get,
}

#[tokio::test]
async fn test_stateful_behavior_creation() {
    let _behavior: Behavior<TestMsg> = stateful(0i32, |count, msg, _ctx| match msg {
        TestMsg::Ping => BehaviorAction::Same,
        TestMsg::Add(n) => {
            *count += n;
            BehaviorAction::Same
        }
        TestMsg::Get => BehaviorAction::Same,
    });
}

#[tokio::test]
async fn test_stateless_behavior_creation() {
    let _behavior: Behavior<TestMsg> = stateless(|msg, _ctx| {
        Box::pin(async move {
            match msg {
                TestMsg::Ping => BehaviorAction::Same,
                TestMsg::Add(_) => BehaviorAction::stop(),
                TestMsg::Get => BehaviorAction::Same,
            }
        })
    });
}

#[tokio::test]
async fn test_behavior_action_stop_helpers() {
    let stop_none = BehaviorAction::<()>::stop();
    assert!(stop_none.is_stop());
    let stop_reason = BehaviorAction::<()>::stop_with_reason("done");
    assert!(stop_reason.is_stop());
    assert!(matches!(stop_reason, BehaviorAction::Stop(Some(r)) if r == "done"));
}

#[tokio::test]
async fn test_behavior_wrapper_receive_same() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let counter = stateful(0i32, |count, msg: TestMsg, _ctx| match msg {
        TestMsg::Add(n) => {
            *count += n;
            BehaviorAction::Same
        }
        _ => BehaviorAction::Same,
    });
    let wrapper: BehaviorWrapper<TestMsg> = counter.into_actor();
    let actor_ref = system.spawn(wrapper).await.unwrap();
    actor_ref.tell(TestMsg::Add(10)).await.unwrap();
    actor_ref.tell(TestMsg::Add(5)).await.unwrap();
    let _: () = actor_ref.ask(TestMsg::Get).await.unwrap();
    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_behavior_wrapper_pack_unpack() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let echo = stateless(|msg: TestMsg, _ctx| {
        Box::pin(async move {
            let _ = msg;
            BehaviorAction::Same
        })
    });
    let wrapper: BehaviorWrapper<TestMsg> = echo.into_actor();
    let _ref = system.spawn(wrapper).await.unwrap();
    system.shutdown().await.unwrap();
}
