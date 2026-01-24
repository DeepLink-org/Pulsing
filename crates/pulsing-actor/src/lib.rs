//! # Pulsing Actor System
//!
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
//!
//! Lightweight distributed actor framework (gossip discovery, HTTP/2 transport).
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use pulsing_actor::prelude::*;
//!
//! #[derive(Serialize, Deserialize)]
//! struct Ping { value: i32 }
//! #[derive(Serialize, Deserialize)]
//! struct Pong { result: i32 }
//!
//! struct Counter { count: i32 }
//!
//! #[async_trait]
//! impl Actor for Counter {
//!     async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext)
//!         -> anyhow::Result<Message>
//!     {
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
//!     let system = ActorSystem::builder().build().await?;
//!     let actor_ref = system.spawn_named("services/counter", Counter { count: 0 }).await?;
//!     let pong: Pong = actor_ref.ask(Ping { value: 42 }).await?;
//!     println!("Result: {}", pong.result);
//!     system.shutdown().await?;
//!     Ok(())
//! }
//! ```

pub mod actor;
pub mod behavior;
pub mod circuit_breaker;
pub mod cluster;
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
        ActorSystem, ActorSystemAdvancedExt, ActorSystemCoreExt, ActorSystemOpsExt, ResolveOptions,
        SpawnOptions, SystemConfig,
    };
    pub use async_trait::async_trait;
    pub use serde::{Deserialize, Serialize};
}

pub use prelude::*;
