//! # Pulsing Actor System
//!
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
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
//! // Define messages
//! #[derive(Serialize, Deserialize)]
//! struct Ping { value: i32 }
//!
//! #[derive(Serialize, Deserialize)]
//! struct Pong { result: i32 }
//!
//! // Define an actor - no boilerplate!
//! struct Counter { count: i32 }
//!
//! #[async_trait]
//! impl Actor for Counter {
//!     async fn receive(
//!         &mut self,
//!         msg: Message,
//!         ctx: &mut ActorContext,
//!     ) -> anyhow::Result<Message> {
//!         if msg.msg_type().ends_with("Ping") {
//!             let ping: Ping = msg.unpack()?;
//!             self.count += ping.value;
//!             return Message::pack(&Pong { result: self.count });
//!         }
//!         Err(anyhow::anyhow!("Unknown message"))
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let system = ActorSystem::new(SystemConfig::standalone()).await?;
//!
//!     // Spawn with a name - system assigns the ID
//!     let actor_ref = system.spawn("counter", Counter { count: 0 }).await?;
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
pub mod circuit_breaker;
pub mod cluster;
pub mod metrics;
pub mod policies;
pub mod supervision;
pub mod system;
pub mod system_actor;
pub mod tracing;
pub mod transport;
pub mod watch;

/// Prelude - commonly used types
///
/// Import with: `use pulsing_actor::prelude::*;`
///
/// Contains only the essentials for building actors:
/// - `Actor` - The trait to implement
/// - `ActorContext` - Passed to handlers, use `ctx.id()` to get actor ID
/// - `ActorRef` - Handle to send messages
/// - `Message` - The message type
/// - `ActorSystem`, `SystemConfig` - System setup
/// - `async_trait`, `Serialize`, `Deserialize` - Re-exports
///
/// For advanced usage (ActorPath, ActorAddress, NodeId, etc.),
/// import from `pulsing_actor::actor::*`.
pub mod prelude {
    pub use crate::actor::{Actor, ActorContext, ActorRef, Message};
    pub use crate::supervision::{BackoffStrategy, RestartPolicy, SupervisionSpec};
    pub use crate::system::{ActorSystem, ResolveOptions, SpawnOptions, SystemConfig};
    pub use async_trait::async_trait;
    pub use serde::{Deserialize, Serialize};
}

pub use prelude::*;
