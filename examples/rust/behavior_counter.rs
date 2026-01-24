//! Behavior-based Actor - Minimal Example
//!
//! Run: cargo run --example behavior_counter

use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};
use pulsing_actor::prelude::*;

fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n, _ctx| {
        *count += n;
        println!("count = {}", *count);
        BehaviorAction::Same
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;

    // Behavior implements IntoActor, can be passed directly to spawn_named
    let counter_ref = system.spawn_named("actors/counter", counter(0)).await?;

    counter_ref.tell(5).await?;
    counter_ref.tell(3).await?;
    counter_ref.tell(-2).await?;

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    system.shutdown().await
}
