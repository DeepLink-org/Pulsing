//! Ping-Pong Example - Simplest Actor Communication
//!
//! Run: cargo run --example ping_pong -p pulsing-actor

use pulsing_actor::prelude::*;

struct Echo;

#[async_trait]
impl Actor for Echo {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let s: String = msg.unpack()?;
        Message::pack(&format!("echo: {}", s))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::builder().build().await?;
    let echo = system.spawn("echo", Echo).await?;

    // ask: send and wait for response
    let resp: String = echo.ask("hello".to_string()).await?;
    println!("{}", resp);

    // tell: fire and forget
    echo.tell("world".to_string()).await?;

    system.shutdown().await
}
