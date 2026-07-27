use thiserror::Error;

use super::{SessionId, TurnId};

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ForgeProtocolError {
    #[error("session not found: {0}")]
    SessionNotFound(SessionId),
    #[error("session already exists: {0}")]
    SessionAlreadyExists(SessionId),
    #[error("session is closed: {0}")]
    SessionClosed(SessionId),
    #[error("session already has an active turn: {0}")]
    SessionBusy(SessionId),
    #[error("turn is not active: {0}")]
    TurnNotActive(TurnId),
    #[error("event sequence conflict: expected {expected}, actual {actual}")]
    SequenceConflict { expected: u64, actual: u64 },
    #[error("invalid state transition: {0}")]
    InvalidTransition(String),
    #[error("event subscription lagged by {0} events")]
    SubscriptionLagged(u64),
    #[error("event subscription closed")]
    SubscriptionClosed,
    #[error("invalid protocol: expected {expected}, got {actual}")]
    InvalidProtocol {
        expected: &'static str,
        actual: String,
    },
    #[error("unsupported {protocol} protocol major version {major}")]
    UnsupportedVersion { protocol: String, major: u16 },
    #[error("agent task failed: {0}")]
    Agent(String),
    #[error("internal Forge error: {0}")]
    Internal(String),
}
