//! Cluster example demonstrating multi-node actor communication
//!
//! This example shows how to:
//! 1. Start multiple actor system nodes
//! 2. Join them into a cluster using gossip protocol
//! 3. Communicate with actors across nodes
//!
//! Run two instances in separate terminals:
//!   Terminal 1: cargo run --example cluster -p pulsing-actor -- --node 1
//!   Terminal 2: cargo run --example cluster -p pulsing-actor -- --node 2

use pulsing_actor::prelude::*;
use std::time::Duration;

// Define messages
#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetCount;

impl Message for GetCount {
    fn type_id() -> &'static str {
        "GetCount"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Increment {
    amount: i32,
}

impl Message for Increment {
    fn type_id() -> &'static str {
        "Increment"
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CountResponse {
    count: i32,
    node: String,
}

impl Message for CountResponse {
    fn type_id() -> &'static str {
        "CountResponse"
    }
}

// Shared counter actor
struct SharedCounter {
    id: ActorId,
    count: i32,
    node_name: String,
}

#[async_trait]
impl Actor for SharedCounter {
    fn id(&self) -> &ActorId {
        &self.id
    }

    async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!(
            "[{}] SharedCounter started with count: {}",
            self.node_name, self.count
        );
        Ok(())
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "GetCount" => {
                println!(
                    "[{}] GetCount request, current: {}",
                    self.node_name, self.count
                );
                RawMessage::from_message(&CountResponse {
                    count: self.count,
                    node: self.node_name.clone(),
                })
            }
            "Increment" => {
                let inc: Increment = msg.into_message()?;
                self.count += inc.amount;
                println!(
                    "[{}] Incremented by {}, new count: {}",
                    self.node_name, inc.amount, self.count
                );
                RawMessage::from_message(&CountResponse {
                    count: self.count,
                    node: self.node_name.clone(),
                })
            }
            _ => Err(anyhow::anyhow!("Unknown message type: {}", msg.msg_type)),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,pulsing_actor=debug")
        .init();

    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    let node_num = if args.len() > 2 && args[1] == "--node" {
        args[2].parse::<u32>().unwrap_or(1)
    } else {
        1
    };

    println!("=== Cluster Actor Example - Node {} ===\n", node_num);

    // Configure based on node number (single HTTP port per node)
    let (port, seed_nodes) = match node_num {
        1 => (8001, vec![]),
        2 => (8002, vec!["127.0.0.1:8001".parse().unwrap()]),
        _ => (8000 + node_num as u16, vec![]),
    };

    let addr = format!("127.0.0.1:{}", port).parse()?;
    let config = SystemConfig::with_addr(addr).with_seeds(seed_nodes);

    let system = ActorSystem::new(config).await?;
    println!("Node {} started at {}", node_num, system.addr());

    // Node 1: Create the shared counter actor
    if node_num == 1 {
        let counter = SharedCounter {
            id: ActorId::local("shared-counter"),
            count: 0,
            node_name: format!("node-{}", node_num),
        };

        let _actor_ref = system.spawn(counter).await?;
        println!("Created shared-counter actor on node 1");

        // Keep node 1 running
        println!("\nNode 1 is running. Press Ctrl+C to stop.\n");
        println!("Start node 2 in another terminal to test cross-node communication.");

        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let members = system.members().await;
            println!(
                "Cluster members: {} (local actors: {:?})",
                members.len(),
                system.local_actor_names()
            );
        }
    }

    // Node 2: Connect and interact with the remote actor
    if node_num == 2 {
        println!("Waiting for cluster to sync...");
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Show cluster members
        let members = system.members().await;
        println!("Cluster members: {:?}", members.len());

        // Try to get reference to remote actor
        let actor_id = ActorId::new(NodeId::new(""), "shared-counter".to_string());

        // Wait for the actor to be discoverable
        println!("Looking for shared-counter actor...");
        let mut attempts = 0;
        let actor_ref = loop {
            attempts += 1;
            match system.actor_ref(&actor_id).await {
                Ok(r) => break r,
                Err(_) if attempts < 10 => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                }
                Err(e) => {
                    println!("Failed to find actor after {} attempts: {}", attempts, e);
                    println!("Make sure node 1 is running!");
                    system.shutdown().await?;
                    return Ok(());
                }
            }
        };

        println!("Found actor: {:?}", actor_ref.id());
        println!("Is local: {}", actor_ref.is_local());

        // Interact with the remote actor
        println!("\n--- Sending messages to remote actor ---\n");

        // Get current count
        let response: CountResponse = actor_ref.ask(GetCount).await?;
        println!("Current count: {} (from {})", response.count, response.node);

        // Increment a few times
        for i in 1..=3 {
            let response: CountResponse = actor_ref.ask(Increment { amount: i * 10 }).await?;
            println!(
                "After increment {}: count = {} (from {})",
                i * 10,
                response.count,
                response.node
            );
        }

        // Final count
        let response: CountResponse = actor_ref.ask(GetCount).await?;
        println!("\nFinal count: {} (from {})", response.count, response.node);

        println!("\nNode 2 finished. Shutting down...");
        system.shutdown().await?;
    }

    Ok(())
}
