//! # Pulsing Actor System
//!
//! A lightweight, zero-external-dependency distributed actor framework.
//!
//! ## Features
//!
//! - **Zero external dependencies**: No etcd, nats, or redis required
//! - **Gossip-based discovery**: Automatic cluster membership using SWIM protocol
//! - **Location-transparent ActorRef**: Same API for local and remote actors
//! - **Async/await native**: Built on tokio
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         ActorSystem                              │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
//! │  │  Actor 1    │  │  Actor 2    │  │      Cluster Module      │  │
//! │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────────────────┐  │  │
//! │  │  │Mailbox│  │  │  │Mailbox│  │  │  │  Gossip Protocol  │  │  │
//! │  │  └───────┘  │  │  └───────┘  │  │  │  (SWIM-like)      │  │  │
//! │  └─────────────┘  └─────────────┘  │  └───────────────────┘  │  │
//! │         ↑               ↑          │           ↑              │  │
//! │         └───────┬───────┘          │           │              │  │
//! │                 │                  │           │              │  │
//! │        ┌────────┴────────┐         │  ┌────────┴────────┐    │  │
//! │        │  Actor Registry │←────────┼──│  Member Registry │    │  │
//! │        └─────────────────┘         │  └─────────────────┘    │  │
//! │                                    └─────────────────────────┘  │
//! │                          ↕ TCP Transport                         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use pulsing_actor::prelude::*;
//!
//! // Define a message
//! #[derive(Serialize, Deserialize)]
//! struct Ping { value: i32 }
//!
//! impl Message for Ping {
//!     fn type_id() -> &'static str { "Ping" }
//! }
//!
//! #[derive(Serialize, Deserialize)]
//! struct Pong { result: i32 }
//!
//! impl Message for Pong {
//!     fn type_id() -> &'static str { "Pong" }
//! }
//!
//! // Define an actor
//! struct CounterActor {
//!     id: ActorId,
//!     count: i32,
//! }
//!
//! #[async_trait]
//! impl Actor for CounterActor {
//!     fn id(&self) -> &ActorId { &self.id }
//!
//!     async fn receive(
//!         &mut self,
//!         msg: RawMessage,
//!         _ctx: &mut ActorContext,
//!     ) -> anyhow::Result<RawMessage> {
//!         match msg.msg_type.as_str() {
//!             "Ping" => {
//!                 let ping: Ping = msg.into_message()?;
//!                 self.count += ping.value;
//!                 RawMessage::from_message(&Pong { result: self.count })
//!             }
//!             _ => Err(anyhow::anyhow!("Unknown message"))
//!         }
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create actor system
//!     let system = ActorSystem::new(SystemConfig::standalone()).await?;
//!
//!     // Spawn actor
//!     let actor = CounterActor {
//!         id: ActorId::local("counter"),
//!         count: 0,
//!     };
//!     let actor_ref = system.spawn(actor).await?;
//!
//!     // Send message and get response
//!     let pong: Pong = actor_ref.ask(Ping { value: 42 }).await?;
//!     println!("Result: {}", pong.result);
//!
//!     system.shutdown().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Cluster Mode
//!
//! ```rust,ignore
//! // Node 1 - Start first node
//! let config = SystemConfig::with_addrs(
//!     "0.0.0.0:8000".parse()?,  // TCP
//!     "0.0.0.0:7000".parse()?,  // Gossip
//! );
//! let system1 = ActorSystem::new(config).await?;
//!
//! // Node 2 - Join existing cluster
//! let config = SystemConfig::with_addrs(
//!     "0.0.0.0:8001".parse()?,
//!     "0.0.0.0:7001".parse()?,
//! ).with_seeds(vec!["192.168.1.100:7000".parse()?]);
//!
//! let system2 = ActorSystem::new(config).await?;
//!
//! // Get reference to actor on another node
//! let remote_ref = system2.actor_ref(&actor_id).await?;
//! let result: Pong = remote_ref.ask(Ping { value: 10 }).await?;
//! ```

pub mod actor;
pub mod cluster;
pub mod system;
pub mod transport;

/// Prelude - commonly used types
pub mod prelude {
    pub use crate::actor::{
        Actor, ActorContext, ActorId, ActorRef, Handler, Message, MessageHandler, NodeId,
        RawMessage,
    };
    pub use crate::cluster::{GossipCluster, GossipConfig, MemberInfo, MemberStatus};
    pub use crate::system::{ActorSystem, SystemConfig};
    pub use crate::transport::TcpTransport;

    pub use async_trait::async_trait;
    pub use serde::{de::DeserializeOwned, Deserialize, Serialize};
}

pub use prelude::*;
