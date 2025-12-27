//! Ping-Pong Example - Basic Actor Communication
//!
//! Demonstrates:
//! - Actor lifecycle (on_start, on_stop)
//! - Request-response (ask) and fire-and-forget (tell) patterns
//!
//! Run: cargo run --example ping_pong -p pulsing-actor

use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};

// Messages
#[derive(Serialize, Deserialize, Debug)]
struct Ping(i32);

#[derive(Serialize, Deserialize, Debug)]
struct Pong(i32);

#[derive(Serialize, Deserialize, Debug)]
struct GetCount;

#[derive(Serialize, Deserialize, Debug)]
struct Count(i32);

// Actor
struct Counter {
    count: i32,
}

#[async_trait]
impl Actor for Counter {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!("[{}] Started with count: {}", ctx.id(), self.count);
        Ok(())
    }

    async fn on_stop(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!("[{}] Stopped with count: {}", ctx.id(), self.count);
        Ok(())
    }

    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> anyhow::Result<Message> {
        match msg.msg_type() {
            t if t.ends_with("Ping") => {
                let Ping(value) = msg.unpack()?;
                self.count += value;
                println!("[{}] Ping({}) -> count = {}", ctx.id(), value, self.count);
                Message::pack(&Pong(self.count))
            }
            t if t.ends_with("GetCount") => Message::pack(&Count(self.count)),
            _ => Err(anyhow::anyhow!("Unknown: {}", msg.msg_type())),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Pulsing Ping-Pong Example ===\n");

    // Create system and spawn actor
    let system = ActorSystem::new(SystemConfig::standalone()).await?;
    let actor = system.spawn("counter", Counter { count: 0 }).await?;
    println!("✓ System started, actor spawned\n");

    // Request-Response (ask)
    println!("--- Request-Response (ask) ---");
    for i in 1..=3 {
        let Pong(result) = actor.ask(Ping(i * 10)).await?;
        println!("Ping({}) -> Pong({})", i * 10, result);
    }

    // Fire-and-Forget (tell)
    println!("\n--- Fire-and-Forget (tell) ---");
    actor.tell(Ping(100)).await?;
    println!("Sent Ping(100) without waiting");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let Count(final_count) = actor.ask(GetCount).await?;
    println!("Final count: {}\n", final_count);

    system.shutdown().await
}
