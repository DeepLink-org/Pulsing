//! Actor module - core actor abstractions

mod context;
mod mailbox;
mod reference;
mod traits;

pub use context::{ActorContext, ActorSystemRef};
pub use mailbox::{Envelope, Mailbox, MailboxSender, DEFAULT_MAILBOX_SIZE};
pub use reference::{ActorRef, ActorRefInner, RemoteActorRef, RemoteTransport};
pub use traits::{Actor, ActorId, Handler, Message, MessageHandler, NodeId, RawMessage};
