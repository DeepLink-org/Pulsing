//! Message Patterns Example
//!
//! Three core patterns:
//! 1. RPC (Single -> Single)
//! 2. Server Streaming (Single -> Stream)
//! 3. Client Streaming (Stream -> Single)
//!
//! Run: cargo run --example message_patterns -p pulsing-actor

use pulsing_actor::prelude::*;
use tokio_stream::StreamExt;

struct Demo;

#[async_trait]
impl Actor for Demo {
    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        match msg.msg_type() {
            // Pattern 1: RPC - String in, String out
            "echo" => {
                let s: String = msg.unpack()?;
                Message::pack(&format!("Hello, {}!", s))
            }

            // Pattern 2: Server Streaming - return stream of i32
            "count" => {
                let n: i32 = msg.unpack()?;
                let (tx, rx) = tokio::sync::mpsc::channel(32);
                tokio::spawn(async move {
                    for i in 1..=n {
                        let _ = tx.send(Ok(Message::pack(&i).unwrap())).await;
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                });
                Ok(Message::from_channel("i32", rx))
            }

            // Pattern 3: Client Streaming - sum stream of i32
            "sum" => {
                let Message::Stream { mut stream, .. } = msg else {
                    return Err(pulsing_actor::error::PulsingError::from(
                        pulsing_actor::error::RuntimeError::Other("Expected stream".into()),
                    ));
                };
                let mut total = 0i32;
                while let Some(chunk) = stream.next().await {
                    let n: i32 = chunk?.unpack()?;
                    total += n;
                }
                Message::pack(&total)
            }

            _ => Err(pulsing_actor::error::PulsingError::from(
                pulsing_actor::error::RuntimeError::Other(format!("Unknown: {}", msg.msg_type())),
            )),
        }
    }
}

#[tokio::main]
async fn main() -> pulsing_actor::error::Result<()> {
    println!("=== Message Patterns ===\n");

    let system = ActorSystem::builder().build().await?;
    let actor = system.spawn_named("test/demo", Demo).await?;

    // Pattern 1: RPC
    println!("--- RPC ---");
    let resp: String = actor.ask("Pulsing".to_string()).await?;
    println!("{}\n", resp);

    // Pattern 2: Server Streaming
    println!("--- Server Streaming ---");
    let req = Message::single(
        "count",
        bincode::serialize(&3i32).map_err(|e| {
            pulsing_actor::error::PulsingError::from(
                pulsing_actor::error::RuntimeError::Serialization(e.to_string()),
            )
        })?,
    );
    let Message::Stream { mut stream, .. } = actor.send(req).await? else {
        return Err(pulsing_actor::error::PulsingError::from(
            pulsing_actor::error::RuntimeError::Other("Expected stream".into()),
        ));
    };
    while let Some(chunk) = stream.next().await {
        let n: i32 = chunk?.unpack()?;
        println!("Received: {}", n);
    }

    // Pattern 3: Client Streaming
    println!("\n--- Client Streaming ---");
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    tokio::spawn(async move {
        for v in [10, 20, 30] {
            let _ = tx.send(Ok(Message::pack(&v).unwrap())).await;
        }
    });
    let total: i32 = actor
        .send(Message::from_channel("sum", rx))
        .await?
        .unpack()?;
    println!("Sum: {}\n", total);

    system.shutdown().await
}
