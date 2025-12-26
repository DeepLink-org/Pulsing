//! Example demonstrating three core messaging patterns in Pulsing:
//! 1. Single Request -> Single Response (RPC)
//! 2. Single Request -> Stream Response (Server Streaming)
//! 3. Stream Request -> Single Response (Client Streaming)
//!
//! Run with: cargo run --example message_patterns -p pulsing-actor

use pulsing_actor::prelude::*;
use tokio_stream::StreamExt;
use serde::{Deserialize, Serialize};

// --- Pattern 1: Single -> Single ---
#[derive(Serialize, Deserialize, Debug)]
struct GreetRequest { name: String }
#[derive(Serialize, Deserialize, Debug)]
struct GreetResponse { message: String }

// --- Pattern 2: Single -> Stream ---
#[derive(Serialize, Deserialize, Debug)]
struct CountRequest { up_to: i32 }
#[derive(Serialize, Deserialize, Debug)]
struct CountItem { value: i32 }

// --- Pattern 3: Stream -> Single ---
#[derive(Serialize, Deserialize, Debug)]
struct SumItem { value: i32 }
#[derive(Serialize, Deserialize, Debug)]
struct SumResponse { total: i32 }

struct PatternDemoActor;

#[async_trait]
impl Actor for PatternDemoActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        match msg.msg_type() {
            // Case 1: Single -> Single
            // Standard request-response pattern.
            t if t.ends_with("GreetRequest") => {
                let req: GreetRequest = msg.unpack()?;
                println!("[Actor] Received GreetRequest: {}", req.name);
                Message::pack(&GreetResponse { message: format!("Hello, {}!", req.name) })
            }

            // Case 2: Single -> Stream (Server Streaming)
            // Useful for scenarios like LLM token generation or progress updates.
            t if t.ends_with("CountRequest") => {
                let req: CountRequest = msg.unpack()?;
                println!("[Actor] Received CountRequest: up_to {}", req.up_to);
                
                let (tx, rx) = tokio::sync::mpsc::channel(32);
                tokio::spawn(async move {
                    for i in 1..=req.up_to {
                        let item = CountItem { value: i };
                        let data = bincode::serialize(&item).unwrap();
                        if tx.send(Ok(data)).await.is_err() {
                            break; // Client stopped listening
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                });
                
                Ok(Message::from_channel("CountItem", rx))
            }

            // Case 3: Stream -> Single (Client Streaming)
            // Useful for uploading data or streaming parameters to an actor.
            t if t.ends_with("StreamRequest") => {
                println!("[Actor] Received StreamRequest (Client Streaming)");
                let mut stream = match msg {
                    Message::Stream { stream, .. } => stream,
                    _ => return Err(anyhow::anyhow!("Expected stream")),
                };
                
                let mut total = 0;
                while let Some(chunk) = stream.next().await {
                    let item: SumItem = bincode::deserialize(&chunk?)?;
                    println!("[Actor]   Received SumItem: {}", item.value);
                    total += item.value;
                }
                
                Message::pack(&SumResponse { total })
            }

            _ => Err(anyhow::anyhow!("Unknown message type: {}", msg.msg_type())),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Pulsing Messaging Patterns Example ===\n");

    // Initialize system
    let system = ActorSystem::new(SystemConfig::standalone()).await?;
    let actor = system.spawn("demo", PatternDemoActor).await?;

    // --- Pattern 1: Single Request -> Single Response (RPC) ---
    println!("--- Pattern 1: RPC ---");
    // Using high-level .ask() which handles packing/unpacking
    let response: GreetResponse = actor.ask(GreetRequest { name: "Pulsing".into() }).await?;
    println!("Client: Got response: {}\n", response.message);


    // --- Pattern 2: Single Request -> Stream Response (Server Streaming) ---
    println!("--- Pattern 2: Server Streaming ---");
    // We use .send() to get the raw Message, which contains the stream
    let req_msg = Message::pack(&CountRequest { up_to: 3 })?;
    let res_msg = actor.send(req_msg).await?;
    
    if let Message::Stream { mut stream, .. } = res_msg {
        while let Some(chunk) = stream.next().await {
            let item: CountItem = bincode::deserialize(&chunk?)?;
            println!("Client: Got stream item: {}", item.value);
        }
    }
    println!();


    // --- Pattern 3: Stream Request -> Single Response (Client Streaming) ---
    println!("--- Pattern 3: Client Streaming ---");
    // Create a channel and wrap it in Message::Stream
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    tokio::spawn(async move {
        for val in [10, 20, 30] {
            let item = SumItem { value: val };
            let data = bincode::serialize(&item).unwrap();
            tx.send(Ok(data)).await.unwrap();
        }
        // Channel drop signifies end of stream
    });
    
    let stream_msg = Message::from_channel("StreamRequest", rx);
    let response_msg = actor.send(stream_msg).await?;
    let final_res: SumResponse = response_msg.unpack()?;
    println!("Client: Final sum response: {}\n", final_res.total);

    println!("All patterns demonstrated successfully.");
    system.shutdown().await
}

