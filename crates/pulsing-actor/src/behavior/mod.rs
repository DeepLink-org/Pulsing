//! Behavior-based actor programming model.
//!
//! This module provides a functional, closure-based API for defining actors,
//! as an alternative to implementing the [`Actor`] trait directly.
//!
//! # Overview
//!
//! The Behavior API is ideal for:
//! - Simple stateful actors with clear state transitions
//! - Quick prototyping without defining new types
//! - Functional programming style
//!
//! # Key Types
//!
//! - [`stateful`] - Create a behavior with mutable state
//! - [`stateless`] - Create a behavior without state
//! - [`BehaviorAction`] - Control flow (continue, stop, become)
//!
//! # Examples
//!
//! ## Simple Counter (Stateful)
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//! use pulsing_actor::behavior::{stateful, BehaviorAction};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, Debug, Clone)]
//! enum CounterMsg {
//!     Increment(i32),
//!     Decrement(i32),
//!     Get,
//! }
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct Count(i32);
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let system = ActorSystem::new(SystemConfig::standalone()).await?;
//!
//! // Create a stateful behavior
//! let counter = stateful(0i32, |count, msg: CounterMsg, _ctx| {
//!     match msg {
//!         CounterMsg::Increment(n) => {
//!             *count += n;
//!             BehaviorAction::Same
//!         }
//!         CounterMsg::Decrement(n) => {
//!             *count -= n;
//!             BehaviorAction::Same
//!         }
//!         CounterMsg::Get => {
//!             println!("Current count: {}", count);
//!             BehaviorAction::Same
//!         }
//!     }
//! });
//!
//! let actor = system.spawn(counter).await?;
//! actor.tell(CounterMsg::Increment(10)).await?;
//! actor.tell(CounterMsg::Decrement(3)).await?;
//! actor.tell(CounterMsg::Get).await?;
//!
//! system.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Stateless Echo Server
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//! use pulsing_actor::behavior::{stateless, BehaviorAction};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, Debug, Clone)]
//! struct Echo { text: String }
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct EchoReply { text: String }
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let system = ActorSystem::new(SystemConfig::standalone()).await?;
//!
//! // Create a stateless behavior
//! let echo = stateless(|msg: Echo, _ctx| {
//!     Box::pin(async move {
//!         let reply = EchoReply { text: msg.text.clone() };
//!         // In real code, you would send the reply back
//!         let _ = reply;
//!         BehaviorAction::Same
//!     })
//! });
//!
//! let actor = system.spawn(echo).await?;
//! actor.tell(Echo { text: "Hello!".into() }).await?;
//!
//! system.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## State Machine with Become
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//! use pulsing_actor::behavior::{stateful, BehaviorAction};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, Debug, Clone)]
//! enum TrafficLightMsg {
//!     Next,
//!     GetState,
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! # let system = ActorSystem::new(SystemConfig::standalone()).await?;
//! // Traffic light state machine
//! let green = stateful((), |_, msg: TrafficLightMsg, ctx| {
//!     match msg {
//!         TrafficLightMsg::Next => {
//!             println!("Green -> Yellow");
//!             // Transition to yellow state
//!             BehaviorAction::Same
//!         }
//!         TrafficLightMsg::GetState => {
//!             println!("State: Green");
//!             BehaviorAction::Same
//!         }
//!     }
//! });
//!
//! let _actor = system.spawn(green).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Stopping a Behavior
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//! use pulsing_actor::behavior::{stateful, BehaviorAction};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, Debug, Clone)]
//! enum WorkerMsg {
//!     DoWork,
//!     Shutdown,
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! # let system = ActorSystem::new(SystemConfig::standalone()).await?;
//! let worker = stateful(0i32, |count, msg: WorkerMsg, _ctx| {
//!     match msg {
//!         WorkerMsg::DoWork => {
//!             *count += 1;
//!             if *count >= 10 {
//!                 println!("Worker completed 10 tasks, stopping...");
//!                 BehaviorAction::stop()
//!             } else {
//!                 BehaviorAction::Same
//!             }
//!         }
//!         WorkerMsg::Shutdown => {
//!             println!("Worker shutting down...");
//!             BehaviorAction::stop_with_reason("Shutdown requested")
//!         }
//!     }
//! });
//!
//! let _actor = system.spawn(worker).await?;
//! # Ok(())
//! # }
//! ```

mod context;
mod core;
mod reference;
mod spawn;

pub use context::BehaviorContext;
pub use core::{stateful, stateless, Behavior, BehaviorAction, BehaviorFn, BehaviorWrapper};
pub use reference::TypedRef;
