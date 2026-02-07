//! Behavior State Machine Example
//!
//! Demonstrates BehaviorAction::Become with state passing.
//! A traffic light that tracks transition count and total cycles.
//!
//! Run: cargo run --example behavior_fsm -p pulsing-actor

use pulsing_actor::behavior::{stateful, Behavior, BehaviorAction};
use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};

/// Statistics passed between states
#[derive(Clone, Debug)]
struct Stats {
    cycles: u32,      // Complete Red->Green->Yellow->Red cycles
    transitions: u32, // Total state transitions
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum Signal {
    Next,  // Advance to next state
    Query, // Print current state and stats
}

fn red(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            println!(
                "[{}] 🔴 Red -> 🟢 Green (transition #{})",
                ctx.name(),
                stats.transitions
            );
            BehaviorAction::Become(green(stats.clone()))
        }
        Signal::Query => {
            println!(
                "[{}] Current: 🔴 Red | cycles: {}, transitions: {}",
                ctx.name(),
                stats.cycles,
                stats.transitions
            );
            BehaviorAction::Same
        }
    })
}

fn green(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            println!(
                "[{}] 🟢 Green -> 🟡 Yellow (transition #{})",
                ctx.name(),
                stats.transitions
            );
            BehaviorAction::Become(yellow(stats.clone()))
        }
        Signal::Query => {
            println!(
                "[{}] Current: 🟢 Green | cycles: {}, transitions: {}",
                ctx.name(),
                stats.cycles,
                stats.transitions
            );
            BehaviorAction::Same
        }
    })
}

fn yellow(stats: Stats) -> Behavior<Signal> {
    stateful(stats, |stats, msg, ctx| match msg {
        Signal::Next => {
            stats.transitions += 1;
            stats.cycles += 1; // Complete one cycle
            println!(
                "[{}] 🟡 Yellow -> 🔴 Red (transition #{}, cycle #{})",
                ctx.name(),
                stats.transitions,
                stats.cycles
            );
            BehaviorAction::Become(red(stats.clone()))
        }
        Signal::Query => {
            println!(
                "[{}] Current: 🟡 Yellow | cycles: {}, transitions: {}",
                ctx.name(),
                stats.cycles,
                stats.transitions
            );
            BehaviorAction::Same
        }
    })
}

#[tokio::main]
async fn main() -> pulsing_actor::error::Result<()> {
    println!("=== Traffic Light State Machine ===\n");

    let system = ActorSystem::builder().build().await?;

    // Start with initial stats
    let initial_stats = Stats {
        cycles: 0,
        transitions: 0,
    };
    // Behavior implements IntoActor, can be passed directly to spawn_named
    let light = system
        .spawn_named("actors/light", red(initial_stats))
        .await?;

    // Run through 2 complete cycles
    for _ in 0..2 {
        light.tell(Signal::Next).await?; // Red -> Green
        light.tell(Signal::Next).await?; // Green -> Yellow
        light.tell(Signal::Next).await?; // Yellow -> Red (cycle complete)
    }

    println!();
    light.tell(Signal::Query).await?; // Check final stats

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    system.shutdown().await
}
