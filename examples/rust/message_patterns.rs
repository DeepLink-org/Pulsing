//! Message Patterns Example
//!
//! Three core patterns:
//! 1. Single -> Single (RPC)
//! 2. Single -> Stream (Server Streaming) - e.g., LLM token generation
//! 3. Stream -> Single (Client Streaming) - e.g., file upload
//!
//! Run: cargo run --example message_patterns -p pulsing-actor

use pulsing_actor::prelude::*;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

// Pattern 1: RPC
#[derive(Serialize, Deserialize, Debug)]
struct Greet {
    name: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Greeting {
    message: String,
}

// Pattern 2: Server Streaming
#[derive(Serialize, Deserialize, Debug)]
struct CountTo {
    n: i32,
}

#[derive(Serialize, Deserialize, Debug)]
struct CountItem {
    value: i32,
}

// Pattern 3: Client Streaming
#[derive(Serialize, Deserialize, Debug)]
struct SumItem {
    value: i32,
}

#[derive(Serialize, Deserialize, Debug)]
struct SumResult {
    total: i32,
}

struct DemoActor;

#[async_trait]
impl Actor for DemoActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        match msg.msg_type() {
            // Pattern 1: RPC - receive request, return response
            t if t.ends_with("Greet") => {
                let req: Greet = msg.unpack()?;
                println!("[Actor] Greet: {}", req.name);
                Message::pack(&Greeting {
                    message: format!("Hello, {}!", req.name),
                })
            }

            // Pattern 2: Server Streaming - return a stream of items
            t if t.ends_with("CountTo") => {
                let req: CountTo = msg.unpack()?;
                println!("[Actor] CountTo: {}", req.n);

                let (tx, rx) = tokio::sync::mpsc::channel(32);
                tokio::spawn(async move {
                    for i in 1..=req.n {
                        let _ = tx
                            .send(Ok(bincode::serialize(&CountItem { value: i }).unwrap()))
                            .await;
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                });
                Ok(Message::from_channel("CountItem", rx))
            }

            // Pattern 3: Client Streaming - receive a stream, return single result
            t if t.ends_with("StreamSum") => {
                println!("[Actor] StreamSum");
                let Message::Stream { mut stream, .. } = msg else {
                    return Err(anyhow::anyhow!("Expected stream"));
                };

                let mut total = 0;
                while let Some(chunk) = stream.next().await {
                    let item: SumItem = bincode::deserialize(&chunk?)?;
                    println!("[Actor]   +{}", item.value);
                    total += item.value;
                }
                Message::pack(&SumResult { total })
            }

            _ => Err(anyhow::anyhow!("Unknown: {}", msg.msg_type())),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Pulsing Message Patterns ===\n");

    let system = ActorSystem::new(SystemConfig::standalone()).await?;
    let actor = system.spawn("demo", DemoActor).await?;

    // Pattern 1: RPC
    println!("--- Pattern 1: RPC ---");
    let resp: Greeting = actor
        .ask(Greet {
            name: "Pulsing".into(),
        })
        .await?;
    println!("Response: {}\n", resp.message);

    // Pattern 2: Server Streaming
    println!("--- Pattern 2: Server Streaming ---");
    let req = Message::pack(&CountTo { n: 3 })?;
    let Message::Stream { mut stream, .. } = actor.send(req).await? else {
        return Err(anyhow::anyhow!("Expected stream"));
    };
    while let Some(chunk) = stream.next().await {
        let item: CountItem = bincode::deserialize(&chunk?)?;
        println!("Received: {}", item.value);
    }

    // Pattern 3: Client Streaming
    println!("\n--- Pattern 3: Client Streaming ---");
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    tokio::spawn(async move {
        for v in [10, 20, 30] {
            let _ = tx
                .send(Ok(bincode::serialize(&SumItem { value: v }).unwrap()))
                .await;
        }
    });
    let resp: SumResult = actor
        .send(Message::from_channel("StreamSum", rx))
        .await?
        .unpack()?;
    println!("Sum: {}\n", resp.total);

    system.shutdown().await
}
