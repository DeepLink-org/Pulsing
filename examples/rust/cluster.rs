//! Cluster Example - Multi-node Communication
//!
//! Demonstrates distributed actors with cross-node messaging.
//!
//! Run in two terminals:
//!   Terminal 1: cargo run --example cluster -p pulsing-actor -- --node 1
//!   Terminal 2: cargo run --example cluster -p pulsing-actor -- --node 2

use pulsing_actor::prelude::*;
use std::time::Duration;

struct Counter {
    count: i32,
    node_id: String,
}

#[async_trait]
impl Actor for Counter {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let n: i32 = msg.unpack()?;
        self.count += n;
        println!("[{}] +{} -> {}", self.node_id, n, self.count);
        Message::pack(&self.count)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let args: Vec<String> = std::env::args().collect();
    let node_num = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("=== Pulsing Cluster - Node {} ===\n", node_num);

    // Node 1: port 8001, no seeds. Node 2+: join via node 1
    let system = match node_num {
        1 => {
            ActorSystem::builder()
                .addr("127.0.0.1:8001")
                .build()
                .await?
        }
        n => {
            ActorSystem::builder()
                .addr(format!("127.0.0.1:{}", 8000 + n as u16))
                .seeds(["127.0.0.1:8001"])
                .build()
                .await?
        }
    };
    println!("✓ Node started: {} @ {}\n", system.node_id(), system.addr());

    let path = "services/counter";

    if node_num == 1 {
        // Node 1: Create actor and wait
        system
            .spawn_named(
                path,
                "counter",
                Counter {
                    count: 0,
                    node_id: system.node_id().to_string(),
                },
            )
            .await?;
        println!("✓ Created named actor: {}", path);
        println!("Start node 2: cargo run --example cluster -p pulsing-actor -- --node 2\n");

        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let members = system.members().await;
            println!("Cluster: {} members", members.len());
            for m in &members {
                println!("  {} @ {} ({:?})", m.node_id, m.addr, m.status);
            }
        }
    } else {
        // Node 2+: Join and interact
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Resolve remote actor
        let actor = loop {
            match system.resolve_named(path, None).await {
                Ok(a) => break a,
                Err(_) => {
                    print!(".");
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        };
        println!("✓ Resolved actor\n");

        // Interact with remote actor
        let count: i32 = actor.ask(0).await?;
        println!("Initial: {}", count);

        for i in 1..=3 {
            let count: i32 = actor.ask(i * 10).await?;
            println!("After +{}: {}", i * 10, count);
        }

        println!("\n✓ Done!");
        system.shutdown().await?;
    }

    Ok(())
}
