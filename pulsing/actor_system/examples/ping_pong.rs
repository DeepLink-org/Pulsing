//! Ping-Pong example demonstrating basic actor communication
//!
//! Run with: cargo run --example ping_pong -p pulsing-actor

use pulsing_actor::prelude::*;

// Messages
#[derive(Serialize, Deserialize, Debug)]
struct Ping(i32);

#[derive(Serialize, Deserialize, Debug)]
struct Pong(i32);

// Actor - minimal definition!
struct Counter {
    count: i32,
}

#[async_trait]
impl Actor for Counter {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!("[{}] started with count: {}", ctx.id().name, self.count);
        Ok(())
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        if msg.msg_type().ends_with("Ping") {
            let Ping(value) = msg.unpack()?;
            self.count += value;
            println!("Ping({}) -> count = {}", value, self.count);
            return Message::pack(&Pong(self.count));
        }
        Err(anyhow::anyhow!("Unknown message"))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Ping-Pong Example ===\n");

    // Create system and spawn actor
    let system = ActorSystem::new(SystemConfig::standalone()).await?;
    let actor = system.spawn("counter", Counter { count: 0 }).await?;

    // Send messages
    for i in 1..=5 {
        let Pong(result) = actor.ask(Ping(i * 10)).await?;
        println!("Got Pong({})\n", result);
    }

    // Fire-and-forget
    actor.tell(Ping(100)).await?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Check final state
    let Pong(final_count) = actor.ask(Ping(0)).await?;
    println!("Final count: {}", final_count);

    system.shutdown().await
}
