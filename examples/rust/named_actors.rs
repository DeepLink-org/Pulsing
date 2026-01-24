//! Named Actors Example - Service Discovery
//!
//! Named actors can be discovered by path (e.g., "services/echo")
//! instead of ActorId, enabling service discovery across cluster.
//!
//! Run: cargo run --example named_actors -p pulsing-actor

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

    // Spawn named actor - name is now the full path
    system.spawn_named("services/echo", Echo).await?;

    // Resolve by name and send message
    let actor = system.resolve("services/echo").await?;
    let resp: String = actor.ask("hello".to_string()).await?;
    println!("{}", resp);

    system.shutdown().await
}
