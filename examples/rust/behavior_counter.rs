//! Behavior-based Actor - Minimal Example
//!
//! Run: cargo run --example behavior_counter

use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction, BehaviorSpawner};
use pulsing_actor::system::ActorSystem;

fn counter(init: i32) -> Behavior<i32> {
    stateful(init, |count, n| {
        *count += n;
        println!("count = {}", *count);
        BehaviorAction::Same
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;
    let counter = system.spawn_behavior("counter", counter(0)).await?;

    counter.tell(5).await?;
    counter.tell(3).await?;
    counter.tell(-2).await?;

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    system.shutdown().await
}
