//! Unified error types for the actor system.

use thiserror::Error;

/// Unified error type for the Pulsing actor system
///
/// This enum encompasses all error categories in the system.
/// It implements `From` for each sub-error type for easy conversion.
#[derive(Error, Debug)]
pub enum PulsingError {
    /// Actor-related errors
    #[error("Actor error: {0}")]
    Actor(#[from] ActorError),

    /// Transport layer errors
    #[error("Transport error: {0}")]
    Transport(#[from] TransportError),

    /// Cluster-related errors
    #[error("Cluster error: {0}")]
    Cluster(#[from] ClusterError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Timeout errors
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Generic errors (for cases not covered by specific types)
    #[error("{0}")]
    Other(String),
}

impl PulsingError {
    /// Create a generic error from a message
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Create a timeout error
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }

    /// Create a serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }
}

impl From<anyhow::Error> for PulsingError {
    fn from(err: anyhow::Error) -> Self {
        // Try to downcast to known error types
        if let Some(actor_err) = err.downcast_ref::<ActorError>() {
            return Self::Actor(actor_err.clone());
        }
        if let Some(transport_err) = err.downcast_ref::<TransportError>() {
            return Self::Transport(transport_err.clone());
        }
        if let Some(cluster_err) = err.downcast_ref::<ClusterError>() {
            return Self::Cluster(cluster_err.clone());
        }
        if let Some(config_err) = err.downcast_ref::<ConfigError>() {
            return Self::Config(config_err.clone());
        }
        Self::Other(err.to_string())
    }
}

/// Actor-related errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ActorError {
    /// Actor not found by name or ID
    #[error("Actor not found: {name}")]
    NotFound { name: String },

    /// Actor already exists with the given name
    #[error("Actor already exists: {name}")]
    AlreadyExists { name: String },

    /// Actor is not local to this node
    #[error("Actor is not local: {name}")]
    NotLocal { name: String },

    /// Actor has stopped and cannot process messages
    #[error("Actor stopped: {name}")]
    Stopped { name: String },

    /// Actor mailbox is full
    #[error("Actor mailbox full: {name}")]
    MailboxFull { name: String },

    /// Invalid actor path format
    #[error("Invalid actor path: {path}")]
    InvalidPath { path: String },

    /// Message type mismatch
    #[error("Message type mismatch: expected {expected}, got {actual}")]
    MessageTypeMismatch { expected: String, actual: String },

    /// Actor spawn failed
    #[error("Failed to spawn actor: {reason}")]
    SpawnFailed { reason: String },
}

impl ActorError {
    /// Create a "not found" error
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::NotFound { name: name.into() }
    }

    /// Create an "already exists" error
    pub fn already_exists(name: impl Into<String>) -> Self {
        Self::AlreadyExists { name: name.into() }
    }

    /// Create a "mailbox full" error
    pub fn mailbox_full(name: impl Into<String>) -> Self {
        Self::MailboxFull { name: name.into() }
    }

    /// Create an "invalid path" error
    pub fn invalid_path(path: impl Into<String>) -> Self {
        Self::InvalidPath { path: path.into() }
    }

    /// Create a "spawn failed" error
    pub fn spawn_failed(reason: impl Into<String>) -> Self {
        Self::SpawnFailed {
            reason: reason.into(),
        }
    }
}

/// Transport layer errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// Connection failed
    #[error("Connection failed to {addr}: {reason}")]
    ConnectionFailed { addr: String, reason: String },

    /// Connection closed unexpectedly
    #[error("Connection closed: {reason}")]
    ConnectionClosed { reason: String },

    /// Request timed out
    #[error("Request timeout after {timeout_ms}ms")]
    RequestTimeout { timeout_ms: u64 },

    /// Invalid response from remote
    #[error("Invalid response: {reason}")]
    InvalidResponse { reason: String },

    /// TLS error
    #[error("TLS error: {reason}")]
    TlsError { reason: String },

    /// Protocol error (HTTP/2)
    #[error("Protocol error: {reason}")]
    ProtocolError { reason: String },
}

impl TransportError {
    /// Create a connection failed error
    pub fn connection_failed(addr: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConnectionFailed {
            addr: addr.into(),
            reason: reason.into(),
        }
    }

    /// Create a request timeout error
    pub fn request_timeout(timeout_ms: u64) -> Self {
        Self::RequestTimeout { timeout_ms }
    }

    /// Create a TLS error
    pub fn tls_error(reason: impl Into<String>) -> Self {
        Self::TlsError {
            reason: reason.into(),
        }
    }
}

/// Cluster-related errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ClusterError {
    /// Cluster not initialized
    #[error("Cluster not initialized")]
    NotInitialized,

    /// Node not found in cluster
    #[error("Node not found: {node_id}")]
    NodeNotFound { node_id: String },

    /// Named actor not found
    #[error("Named actor not found: {path}")]
    NamedActorNotFound { path: String },

    /// No healthy instances available
    #[error("No healthy instances for: {path}")]
    NoHealthyInstances { path: String },

    /// Join failed
    #[error("Failed to join cluster: {reason}")]
    JoinFailed { reason: String },

    /// Gossip protocol error
    #[error("Gossip error: {reason}")]
    GossipError { reason: String },
}

impl ClusterError {
    /// Create a "not initialized" error
    pub fn not_initialized() -> Self {
        Self::NotInitialized
    }

    /// Create a "node not found" error
    pub fn node_not_found(node_id: impl Into<String>) -> Self {
        Self::NodeNotFound {
            node_id: node_id.into(),
        }
    }

    /// Create a "named actor not found" error
    pub fn named_actor_not_found(path: impl Into<String>) -> Self {
        Self::NamedActorNotFound { path: path.into() }
    }

    /// Create a "no healthy instances" error
    pub fn no_healthy_instances(path: impl Into<String>) -> Self {
        Self::NoHealthyInstances { path: path.into() }
    }
}

/// Configuration-related errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// Invalid configuration value
    #[error("Invalid configuration: {field} = {value} ({reason})")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    /// Missing required configuration
    #[error("Missing required configuration: {field}")]
    MissingRequired { field: String },

    /// Conflicting configuration options
    #[error("Conflicting configuration: {reason}")]
    Conflicting { reason: String },

    /// Address parsing error
    #[error("Invalid address '{addr}': {reason}")]
    InvalidAddress { addr: String, reason: String },
}

impl ConfigError {
    /// Create an "invalid value" error
    pub fn invalid_value(
        field: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidValue {
            field: field.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// Create a "missing required" error
    pub fn missing_required(field: impl Into<String>) -> Self {
        Self::MissingRequired {
            field: field.into(),
        }
    }

    /// Create a "conflicting" error
    pub fn conflicting(reason: impl Into<String>) -> Self {
        Self::Conflicting {
            reason: reason.into(),
        }
    }

    /// Create an "invalid address" error
    pub fn invalid_address(addr: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidAddress {
            addr: addr.into(),
            reason: reason.into(),
        }
    }
}

/// Convenience type alias for results using PulsingError
pub type Result<T> = std::result::Result<T, PulsingError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_error_display() {
        let err = ActorError::not_found("my-actor");
        assert!(err.to_string().contains("my-actor"));

        let err = ActorError::already_exists("existing-actor");
        assert!(err.to_string().contains("existing-actor"));
    }

    #[test]
    fn test_transport_error_display() {
        let err = TransportError::connection_failed("127.0.0.1:8000", "connection refused");
        assert!(err.to_string().contains("127.0.0.1:8000"));
        assert!(err.to_string().contains("refused"));

        let err = TransportError::request_timeout(5000);
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_cluster_error_display() {
        let err = ClusterError::not_initialized();
        assert!(err.to_string().contains("not initialized"));

        let err = ClusterError::named_actor_not_found("services/echo");
        assert!(err.to_string().contains("services/echo"));
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::invalid_value("mailbox_capacity", "0", "must be > 0");
        assert!(err.to_string().contains("mailbox_capacity"));

        let err = ConfigError::conflicting("cannot be both head node and worker");
        assert!(err.to_string().contains("head node"));
    }

    #[test]
    fn test_pulsing_error_from_actor_error() {
        let actor_err = ActorError::not_found("test");
        let pulsing_err: PulsingError = actor_err.into();

        assert!(matches!(pulsing_err, PulsingError::Actor(_)));
        assert!(pulsing_err.to_string().contains("test"));
    }

    #[test]
    fn test_pulsing_error_from_transport_error() {
        let transport_err = TransportError::request_timeout(3000);
        let pulsing_err: PulsingError = transport_err.into();

        assert!(matches!(pulsing_err, PulsingError::Transport(_)));
        assert!(pulsing_err.to_string().contains("3000"));
    }

    #[test]
    fn test_pulsing_error_helpers() {
        let err = PulsingError::other("something went wrong");
        assert!(err.to_string().contains("wrong"));

        let err = PulsingError::timeout("operation timed out");
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = ActorError::not_found("test");
        let err2 = ActorError::not_found("test");
        let err3 = ActorError::not_found("other");

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
