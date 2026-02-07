//! Unit tests for BehaviorContext: name, actor_id, self_ref, is_cancelled.

use pulsing_actor::behavior::{stateless, BehaviorAction, BehaviorWrapper};
use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Serialize, Deserialize, Debug)]
enum CtxMsg {
    RecordContext,
    Ping,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct ContextSnapshot {
    name: Option<String>,
    actor_id_str: Option<String>,
    is_cancelled: Option<bool>,
}

#[tokio::test]
async fn test_behavior_context_name_and_actor_id() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let snapshot: Arc<Mutex<ContextSnapshot>> = Arc::new(Mutex::new(ContextSnapshot::default()));
    let snap = snapshot.clone();
    let behavior = stateless(move |msg: CtxMsg, ctx| {
        let snap = snap.clone();
        Box::pin(async move {
            if matches!(msg, CtxMsg::RecordContext) {
                let mut s = snap.lock().await;
                s.name = Some(ctx.name().to_string());
                s.actor_id_str = Some(ctx.actor_id().to_string());
                s.is_cancelled = Some(ctx.is_cancelled());
            }
            BehaviorAction::Same
        })
    });
    let wrapper: BehaviorWrapper<CtxMsg> = behavior.into_actor();
    let actor_ref = system.spawn_named("test/ctx-test", wrapper).await.unwrap();
    actor_ref.tell(CtxMsg::RecordContext).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    let s = snapshot.lock().await;
    assert_eq!(s.name.as_deref(), Some("test/ctx-test"));
    assert!(!s.actor_id_str.as_deref().unwrap_or("").is_empty());
    assert_eq!(s.is_cancelled, Some(false));
    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_behavior_context_self_ref_resolves() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
    let behavior = stateless(|_msg: CtxMsg, ctx| {
        Box::pin(async move {
            let self_ref = ctx.self_ref();
            assert_eq!(self_ref.name(), "test/self-ref-actor");
            let untyped = self_ref.as_untyped();
            assert!(untyped.is_ok());
            BehaviorAction::Same
        })
    });
    let wrapper: BehaviorWrapper<CtxMsg> = behavior.into_actor();
    let actor_ref = system
        .spawn_named("test/self-ref-actor", wrapper)
        .await
        .unwrap();
    let _: () = actor_ref.ask(CtxMsg::Ping).await.unwrap();
    system.shutdown().await.unwrap();
}
