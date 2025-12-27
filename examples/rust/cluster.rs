//! Cluster Example - Multi-node Communication
//!
//! Demonstrates distributed actors with cross-node messaging.
//!
//! Run in two terminals:
//!   Terminal 1: cargo run --example cluster -p pulsing-actor -- --node 1
//!   Terminal 2: cargo run --example cluster -p pulsing-actor -- --node 2

use pulsing_actor::actor::ActorPath;
use pulsing_actor::prelude::*;
use std::time::Duration;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetCount;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Increment(i32);

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CountResponse {
    count: i32,
    from_node: String,
}

struct SharedCounter {
    count: i32,
    node_id: String,
}

#[async_trait]
impl Actor for SharedCounter {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!("[{}] Started on {}", ctx.id(), self.node_id);
        Ok(())
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        match msg.msg_type() {
            t if t.ends_with("GetCount") => Message::pack(&CountResponse {
                count: self.count,
                from_node: self.node_id.clone(),
            }),
            t if t.ends_with("Increment") => {
                let Increment(n) = msg.unpack()?;
                self.count += n;
                println!("[{}] +{} -> {}", self.node_id, n, self.count);
                Message::pack(&CountResponse {
                    count: self.count,
                    from_node: self.node_id.clone(),
                })
            }
            _ => Err(anyhow::anyhow!("Unknown: {}", msg.msg_type())),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let args: Vec<String> = std::env::args().collect();
    let node_num = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("=== Pulsing Cluster - Node {} ===\n", node_num);

    // Node 1: port 8001, no seeds. Node 2+: join via node 1
    let (port, seeds) = match node_num {
        1 => (8001, vec![]),
        _ => (8000 + node_num as u16, vec!["127.0.0.1:8001".parse()?]),
    };

    let config = SystemConfig::with_addr(format!("127.0.0.1:{}", port).parse()?).with_seeds(seeds);
    let system = ActorSystem::new(config).await?;
    println!("✓ Node started: {} @ {}\n", system.node_id(), system.addr());

    let path = ActorPath::new("services/counter").unwrap();

    if node_num == 1 {
        // Node 1: Create actor and wait
        system
            .spawn_named(
                path.clone(),
                "counter",
                SharedCounter {
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
            match system.resolve_named(&path, None).await {
                Ok(a) => break a,
                Err(_) => {
                    print!(".");
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        };
        println!("✓ Resolved actor\n");

        // Interact with remote actor
        let resp: CountResponse = actor.ask(GetCount).await?;
        println!("Initial: {} (from {})", resp.count, resp.from_node);

        for i in 1..=3 {
            let resp: CountResponse = actor.ask(Increment(i * 10)).await?;
            println!(
                "After +{}: {} (from {})",
                i * 10,
                resp.count,
                resp.from_node
            );
        }

        println!("\n✓ Done!");
        system.shutdown().await?;
    }

    Ok(())
}
