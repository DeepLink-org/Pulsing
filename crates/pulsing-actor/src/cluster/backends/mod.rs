//! Naming backend implementations

mod gossip;
mod head;

pub use gossip::GossipBackend;
pub use head::{HeadNodeBackend, HeadNodeConfig, RegisterActorRequest, UnregisterActorRequest};
