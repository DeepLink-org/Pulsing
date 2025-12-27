//! Named Actors Example
//!
//! Named actors can be discovered by service path (e.g., "services/echo")
//! instead of specific ActorId, enabling service discovery and load balancing.
//!
//! Run: cargo run --example named_actors -p pulsing-actor

use pulsing_actor::actor::{ActorAddress, ActorPath};
use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct Echo {
    message: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct EchoResponse {
    echo: String,
    actor_id: String,
}

struct EchoActor;

#[async_trait]
impl Actor for EchoActor {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!("[{}] Started", ctx.id());
        Ok(())
    }

    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let req: Echo = msg.unpack()?;
        println!("[{}] Echo: {}", ctx.id(), req.message);
        Message::pack(&EchoResponse {
            echo: req.message,
            actor_id: ctx.id().to_string(),
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Pulsing Named Actors ===\n");

    let system = ActorSystem::new(SystemConfig::standalone()).await?;
    println!("✓ System started: {}\n", system.node_id());

    // Create named actor with service path
    let path = ActorPath::new("services/echo").unwrap();
    let _actor = system.spawn_named(path.clone(), "echo", EchoActor).await?;
    println!("✓ Created: {} (local name: echo)\n", path);

    // Method 1: Resolve by ActorPath
    println!("--- Resolve by ActorPath ---");
    let resolved = system.resolve_named(&path, None).await?;
    let resp: EchoResponse = resolved
        .ask(Echo {
            message: "Hello!".into(),
        })
        .await?;
    println!("Response: {} (from {})\n", resp.echo, resp.actor_id);

    // Method 2: Resolve by ActorAddress (URI format)
    println!("--- Resolve by ActorAddress ---");
    let addr = ActorAddress::parse("actor:///services/echo")?;
    let resolved2 = system.resolve(&addr).await?;
    let resp2: EchoResponse = resolved2
        .ask(Echo {
            message: "Via URI".into(),
        })
        .await?;
    println!("Response: {} (from {})\n", resp2.echo, resp2.actor_id);

    // List instances
    let instances = system.get_named_instances(&path).await;
    println!("Instances of '{}': {}", path, instances.len());
    for i in &instances {
        println!("  {} @ {} ({:?})", i.node_id, i.addr, i.status);
    }

    println!("\n✓ Done!");
    system.shutdown().await
}
