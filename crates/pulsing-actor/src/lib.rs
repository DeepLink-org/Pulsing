//! # Pulsing Actor System
//!
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
//!
//! Lightweight distributed actor framework (gossip discovery, HTTP/2 transport).
//!
//! ## Quick Start
//!
//! Create your first actor and send messages:
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//! use pulsing_actor::error::PulsingError;
//! use serde::{Deserialize, Serialize};
//!
//! // Define messages
//! #[derive(Serialize, Deserialize, Debug)]
//! struct Ping { value: i32 }
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct Pong { result: i32 }
//!
//! // Define actor state
//! struct Counter { count: i32 }
//!
//! #[async_trait]
//! impl Actor for Counter {
//!     async fn receive(
//!         &mut self,
//!         msg: Message,
//!         _ctx: &mut ActorContext,
//!     ) -> Result<Message, PulsingError> {
//!         if let Ok(ping) = msg.unpack::<Ping>() {
//!             self.count += ping.value;
//!             return Message::pack(&Pong { result: self.count });
//!         }
//!         Err(PulsingError::from(
//!             pulsing_actor::error::RuntimeError::Other("Unknown message type".into())
//!         ))
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create actor system
//!     let system = ActorSystem::new(SystemConfig::standalone()).await?;
//!
//!     // Spawn a named actor
//!     let actor_ref = system
//!         .spawn_named("services/counter", Counter { count: 0 })
//!         .await?;
//!
//!     // Send message and await response
//!     let pong: Pong = actor_ref.ask(Ping { value: 42 }).await?;
//!     println!("Result: {}", pong.result);
//!
//!     // Clean shutdown
//!     system.shutdown().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Using the Behavior API
//!
//! For simpler actors, use the Behavior API with closures:
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//! use pulsing_actor::behavior::stateful;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, Debug, Clone)]
//! enum CounterMsg {
//!     Increment(i32),
//!     Get,
//! }
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct Count(i32);
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let system = ActorSystem::new(SystemConfig::standalone()).await?;
//!
//!     // Create a stateful behavior with closure
//!     let counter = stateful(0i32, |count, msg: CounterMsg, _ctx| {
//!         match msg {
//!             CounterMsg::Increment(n) => {
//!                 *count += n;
//!                 pulsing_actor::behavior::BehaviorAction::Same
//!             }
//!             CounterMsg::Get => {
//!                 // Return current count without changing state
//!                 let _ = count;
//!                 pulsing_actor::behavior::BehaviorAction::Same
//!             }
//!         }
//!     });
//!
//!     let actor_ref = system.spawn(counter).await?;
//!     actor_ref.tell(CounterMsg::Increment(10)).await?;
//!
//!     system.shutdown().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Cluster Mode
//!
//! Run actors across multiple nodes:
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Node 1: Seed node
//!     let addr: std::net::SocketAddr = "0.0.0.0:8000".parse()?;
//!     let config = SystemConfig::with_addr(addr);
//!     let system1 = ActorSystem::new(config).await?;
//!
//!     // Node 2: Join the cluster
//!     let addr: std::net::SocketAddr = "0.0.0.0:8001".parse()?;
//!     let seed: std::net::SocketAddr = "127.0.0.1:8000".parse()?;
//!     let config = SystemConfig::with_addr(addr)
//!         .with_seeds(vec![seed]);
//!     let system2 = ActorSystem::new(config).await?;
//!
//!     // Actors can now communicate across nodes
//!     println!("Cluster formed with 2 nodes");
//!
//!     Ok(())
//! }
//! ```

pub mod actor;
pub mod behavior;
pub mod circuit_breaker;
pub mod cluster;
pub mod connect;
pub mod error;
pub mod metrics;
pub mod policies;
pub mod supervision;
pub mod system;
pub mod system_actor;
pub mod tracing;
pub mod transport;
pub mod watch;

/// Test helpers and macros for writing actor system tests
///
/// This module provides reusable test infrastructure including:
/// - Common test messages (TestPing, TestPong, etc.)
/// - Common test actors (TestEchoActor, TestAccumulatorActor)
/// - Helper functions for test setup
/// - Macros for standardized test patterns
///
/// # Example
/// ```rust,ignore
/// use pulsing_actor::test_helper::*;
/// use pulsing_actor::actor_test;
///
/// actor_test!(test_echo, system, {
///     let echo = spawn_echo_actor(&system, "test/echo").await;
///     let response: TestPong = echo.ask(TestPing { value: 21 }).await.unwrap();
///     assert_eq!(response.result, 42);
/// });
/// ```
#[cfg(any(test, feature = "test-helper"))]
pub mod test_helper;

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
    pub use crate::actor::{Actor, ActorContext, ActorRef, IntoActor, Message};
    pub use crate::supervision::{BackoffStrategy, RestartPolicy, SupervisionSpec};
    pub use crate::system::{
        ActorSystem, ActorSystemCoreExt, ActorSystemOpsExt, ResolveOptions, SpawnOptions,
        SystemConfig,
    };
    pub use async_trait::async_trait;
    pub use serde::{Deserialize, Serialize};
}

pub use prelude::*;
