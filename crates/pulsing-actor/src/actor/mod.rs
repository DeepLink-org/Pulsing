//! Actor module - core actor abstractions

mod address;
mod context;
mod mailbox;
mod reference;
mod traits;

pub use address::{ActorAddress, ActorPath, AddressParseError};
pub use context::{ActorContext, ActorSystemRef};
pub use mailbox::{Envelope, Mailbox, MailboxSender, ResponseChannel, DEFAULT_MAILBOX_SIZE};
pub use reference::{ActorRef, ActorRefInner, RemoteActorRef, RemoteTransport};
pub use traits::{Actor, ActorId, Message, MessageStream, NodeId, PayloadStream, StopReason};
