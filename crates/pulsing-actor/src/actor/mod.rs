//! Actor module - core abstractions.
//!
//! This module provides the fundamental building blocks for the actor system:
//! - [`Actor`] trait - implement this for your actor types
//! - [`ActorRef`] - handle for sending messages to actors
//! - [`ActorContext`] - context passed to actor handlers
//! - [`ActorPath`] and [`ActorAddress`] - actor addressing
//! - [`Message`] - message envelope type
//!
//! # Examples
//!
//! ## Creating a Simple Actor
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//! use pulsing_actor::error::PulsingError;
//! use serde::{Deserialize, Serialize};
//!
//! // Define your messages
//! #[derive(Serialize, Deserialize, Debug)]
//! struct Greet { name: String }
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct Greeting { message: String }
//!
//! // Define your actor
//! struct Greeter;
//!
//! #[async_trait]
//! impl Actor for Greeter {
//!     async fn receive(
//!         &mut self,
//!         msg: Message,
//!         _ctx: &mut ActorContext,
//!     ) -> Result<Message, PulsingError> {
//!         if let Ok(greet) = msg.unpack::<Greet>() {
//!             let response = Greeting {
//!                 message: format!("Hello, {}!", greet.name),
//!             };
//!             return Message::pack(&response);
//!         }
//!         Err(PulsingError::from(
//!             pulsing_actor::error::RuntimeError::Other("Unknown message".into())
//!         ))
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let system = ActorSystem::new(SystemConfig::standalone()).await?;
//! let greeter = system.spawn_named("services/greeter", Greeter).await?;
//!
//! let greeting: Greeting = greeter
//!     .ask(Greet { name: "World".into() })
//!     .await?;
//!
//! println!("{}", greeting.message); // Hello, World!
//! # Ok(())
//! # }
//! ```
//!
//! ## Using Actor Paths
//!
//! ```
//! use pulsing_actor::actor::ActorPath;
//!
//! // Create a path
//! let path = ActorPath::new("services/api/users").unwrap();
//! assert_eq!(path.namespace(), "services");
//! assert_eq!(path.name(), "users");
//!
//! // Get parent path
//! let parent = path.parent().unwrap();
//! assert_eq!(parent.as_str(), "services/api");
//!
//! // Create child path
//! let child = path.child("profile").unwrap();
//! assert_eq!(child.as_str(), "services/api/users/profile");
//! ```
//!
//! ## Handling Different Message Types
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//! use pulsing_actor::error::PulsingError;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! enum CalculatorMsg {
//!     Add(i32),
//!     Subtract(i32),
//!     GetResult,
//! }
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct CountResult(i32);
//!
//! struct Calculator { value: i32 }
//!
//! #[async_trait]
//! impl Actor for Calculator {
//!     async fn receive(
//!         &mut self,
//!         msg: Message,
//!         _ctx: &mut ActorContext,
//!     ) -> Result<Message, PulsingError> {
//!         match msg.unpack::<CalculatorMsg>() {
//!             Ok(CalculatorMsg::Add(n)) => {
//!                 self.value += n;
//!                 Message::pack(&CountResult(self.value))
//!             }
//!             Ok(CalculatorMsg::Subtract(n)) => {
//!                 self.value -= n;
//!                 Message::pack(&CountResult(self.value))
//!             }
//!             Ok(CalculatorMsg::GetResult) => {
//!                 Message::pack(&CountResult(self.value))
//!             }
//!             Err(_e) => Err(PulsingError::from(
//!                 pulsing_actor::error::RuntimeError::Serialization(
//!                     "Failed to unpack message".into()
//!                 )
//!             )),
//!         }
//!     }
//! }
//! ```

mod address;
mod context;
mod mailbox;
mod reference;
mod traits;

pub use address::{ActorAddress, ActorPath, AddressParseError, IntoActorPath};
pub use context::{ActorContext, ActorSystemRef};
pub use mailbox::{Envelope, Mailbox, MailboxSender, ResponseChannel, DEFAULT_MAILBOX_SIZE};
pub use reference::{
    ActorRef, ActorRefInner, ActorResolver, LazyActorRef, RemoteActorRef, RemoteTransport,
};
pub use traits::{Actor, ActorId, IntoActor, Message, MessageStream, NodeId, StopReason};
