//! Persistent Forge sessions, event reduction, and local client.

mod client;
mod reducer;
mod service;
mod store;

pub use client::LocalForgeClient;
pub use reducer::{SessionSnapshot, SessionStatus, TurnSnapshot, TurnStatus};
pub use service::{EventSubscription, ForgeService};
pub use store::{EventStore, InMemoryEventStore};
