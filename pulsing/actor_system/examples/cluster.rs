//! Cluster example - multi-node actor communication
//!
//! Run in two terminals:
//!   Terminal 1: cargo run --example cluster -p pulsing-actor -- --node 1
//!   Terminal 2: cargo run --example cluster -p pulsing-actor -- --node 2

use pulsing_actor::actor::{ActorId, NodeId};
use pulsing_actor::prelude::*;
use std::time::Duration;

// Messages
#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetCount;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Increment(i32);

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CountResponse {
    count: i32,
    node: String,
}

// Actor
struct SharedCounter {
    count: i32,
    node_name: String,
}

#[async_trait]
impl Actor for SharedCounter {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!(
            "[{}] SharedCounter started on {}",
            ctx.id().name,
            self.node_name
        );
        Ok(())
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        if msg.msg_type().ends_with("GetCount") {
            return Message::pack(&CountResponse {
                count: self.count,
                node: self.node_name.clone(),
            });
        }
        if msg.msg_type().ends_with("Increment") {
            let Increment(amount) = msg.unpack()?;
            self.count += amount;
            println!("[{}] count += {} -> {}", self.node_name, amount, self.count);
            return Message::pack(&CountResponse {
                count: self.count,
                node: self.node_name.clone(),
            });
        }
        Err(anyhow::anyhow!("Unknown message"))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,pulsing_actor=debug")
        .init();

    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let node_num = if args.len() > 2 && args[1] == "--node" {
        args[2].parse::<u32>().unwrap_or(1)
    } else {
        1
    };

    println!("=== Cluster Example - Node {} ===\n", node_num);

    // Setup
    let (port, seeds) = match node_num {
        1 => (8001, vec![]),
        _ => (8000 + node_num as u16, vec!["127.0.0.1:8001".parse()?]),
    };

    let config = SystemConfig::with_addr(format!("127.0.0.1:{}", port).parse()?).with_seeds(seeds);
    let system = ActorSystem::new(config).await?;
    println!("Started at {}", system.addr());

    if node_num == 1 {
        // Node 1: Create actor
        system
            .spawn(
                "shared-counter",
                SharedCounter {
                    count: 0,
                    node_name: "node-1".into(),
                },
            )
            .await?;

        println!("Created shared-counter. Waiting for node 2...\n");
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            println!("Members: {}", system.members().await.len());
        }
    } else {
        // Node 2: Connect and interact
        println!("Waiting for cluster sync...");
        tokio::time::sleep(Duration::from_secs(2)).await;

        let actor_id = ActorId::new(NodeId::new(""), "shared-counter");
        let actor = system.actor_ref(&actor_id).await?;

        println!("Found actor, sending messages...\n");

        let resp: CountResponse = actor.ask(GetCount).await?;
        println!("Count: {} (from {})", resp.count, resp.node);

        for i in 1..=3 {
            let resp: CountResponse = actor.ask(Increment(i * 10)).await?;
            println!("After +{}: {} (from {})", i * 10, resp.count, resp.node);
        }

        system.shutdown().await?;
    }

    Ok(())
}
