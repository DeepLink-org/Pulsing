//! Actor module - core actor abstractions

mod address;
mod context;
mod mailbox;
mod reference;
mod traits;

pub use address::{ActorAddress, ActorPath, AddressParseError, LOCALHOST};
pub use context::{ActorContext, ActorSystemRef};
pub use mailbox::{Envelope, EnvelopeResponse, Mailbox, MailboxSender, DEFAULT_MAILBOX_SIZE};
pub use reference::{ActorRef, ActorRefInner, RemoteActorRef, RemoteTransport};
pub use traits::{
    empty_stream, Actor, ActorId, Message, MessageDispatcher, MessageStream, NodeId, PayloadStream,
    StopReason, Terminated,
};
