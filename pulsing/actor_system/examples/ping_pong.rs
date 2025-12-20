//! Ping-Pong example demonstrating basic actor communication
//!
//! Run with: cargo run --example ping_pong -p pulsing-actor

use pulsing_actor::prelude::*;

// Define messages
#[derive(Serialize, Deserialize, Debug)]
struct Ping {
    value: i32,
}

impl Message for Ping {
    fn type_id() -> &'static str {
        "Ping"
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Pong {
    result: i32,
}

impl Message for Pong {
    fn type_id() -> &'static str {
        "Pong"
    }
}

// Define an actor
struct CounterActor {
    id: ActorId,
    count: i32,
}

#[async_trait]
impl Actor for CounterActor {
    fn id(&self) -> &ActorId {
        &self.id
    }

    async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!("CounterActor started with initial count: {}", self.count);
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        println!("CounterActor stopped with final count: {}", self.count);
        Ok(())
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        match msg.msg_type.as_str() {
            "Ping" => {
                let ping: Ping = msg.into_message()?;
                self.count += ping.value;
                println!("Received Ping({}), count is now {}", ping.value, self.count);
                RawMessage::from_message(&Pong { result: self.count })
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

    println!("=== Ping-Pong Actor Example ===\n");

    // Create actor system (standalone mode)
    let config = SystemConfig::standalone();
    let system = ActorSystem::new(config).await?;

    println!("Actor system started at {:?}", system.tcp_addr());

    // Spawn counter actor
    let actor = CounterActor {
        id: ActorId::local("counter"),
        count: 0,
    };

    let actor_ref = system.spawn(actor).await?;
    println!("Spawned actor: {:?}\n", actor_ref.id());

    // Send some ping messages
    for i in 1..=5 {
        let pong: Pong = actor_ref.ask(Ping { value: i * 10 }).await?;
        println!("Got Pong with result: {}\n", pong.result);
    }

    // Fire-and-forget message
    println!("Sending tell (fire-and-forget)...");
    actor_ref.tell(Ping { value: 100 }).await?;

    // Give time for the tell to be processed
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Final ask to see the result
    let final_pong: Pong = actor_ref.ask(Ping { value: 0 }).await?;
    println!("Final count: {}", final_pong.result);

    // Shutdown
    println!("\nShutting down...");
    system.shutdown().await?;

    println!("Done!");
    Ok(())
}

