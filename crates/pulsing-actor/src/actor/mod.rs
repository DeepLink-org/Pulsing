//! Actor module - core abstractions.

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
