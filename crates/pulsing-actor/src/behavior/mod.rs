//! Behavior-based Actor Programming Model
//!
//! This module provides a type-safe, functional actor programming interface
//! inspired by Akka Typed. It shares the underlying ActorSystem infrastructure
//! with the traditional Actor trait but offers a completely new programming model.
//!
//! # Key Concepts
//!
//! - **Behavior<M>**: An actor is defined as a message-handling function
//! - **TypedRef<M>**: Type-safe actor reference with compile-time message checking
//! - **BehaviorAction**: Control actor lifecycle and state transitions
//!
//! # Example
//!
//! ```rust,ignore
//! use pulsing_actor::behavior::*;
//!
//! // Define message type
//! #[derive(Serialize, Deserialize)]
//! enum CounterMsg {
//!     Increment(i32),
//!     Get { reply_to: TypedRef<i32> },
//! }
//!
//! // Define actor as a function returning Behavior
//! fn counter(initial: i32) -> Behavior<CounterMsg> {
//!     stateful(initial, |value, msg, ctx| {
//!         Box::pin(async move {
//!             match msg {
//!                 CounterMsg::Increment(n) => {
//!                     *value += n;
//!                     BehaviorAction::Same
//!                 }
//!                 CounterMsg::Get { reply_to } => {
//!                     let _ = reply_to.tell(*value).await;
//!                     BehaviorAction::Same
//!                 }
//!             }
//!         })
//!     })
//! }
//!
//! // Spawn and use
//! let counter_ref: TypedRef<CounterMsg> = system.spawn_behavior("counter", counter(0)).await?;
//! counter_ref.tell(CounterMsg::Increment(5)).await?;
//! ```

mod context;
mod core;
mod reference;
mod spawn;

pub use context::BehaviorContext;
pub use core::{stateful, stateless, Behavior, BehaviorAction, BehaviorFn};
pub use reference::TypedRef;
pub use spawn::BehaviorSpawner;
