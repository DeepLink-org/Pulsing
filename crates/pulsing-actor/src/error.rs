//! Unified error types for the actor system.
//!
//! Error hierarchy (matches Python exception structure):
//! - PulsingError: Top-level error enum
//!   - RuntimeError: Framework/system-level errors
//!     - Actor system errors (NotFound, Stopped, etc.)
//!     - Transport errors (ConnectionFailed, etc.)
//!     - Cluster errors (NodeNotFound, etc.)
//!     - Config errors (InvalidValue, etc.)
//!     - I/O errors, Serialization errors
//!       → Maps to Python: PulsingRuntimeError
//!   - ActorError: User Actor execution errors
//!     - Business errors (user input errors)
//!     - System errors (internal errors from user code)
//!     - Timeout errors (operation timeouts)
//!     - Unsupported errors (unsupported operations)
//!       → Maps to Python: PulsingActorError (and subclasses)

use thiserror::Error;

/// Unified error type for the Pulsing actor system
///
/// This enum encompasses all error categories in the system.
/// Errors are divided into two main categories:
/// - RuntimeError: Framework/system-level errors
/// - ActorError: User Actor execution errors
#[derive(Error, Debug)]
pub enum PulsingError {
    /// Runtime errors: Framework/system-level errors
    #[error("Runtime error: {0}")]
    Runtime(#[from] RuntimeError),

    /// Actor errors: User Actor execution errors
    #[error("Actor error: {0}")]
    Actor(#[from] ActorError),
}

impl PulsingError {
    /// Check if this is a runtime error
    pub fn is_runtime(&self) -> bool {
        matches!(self, Self::Runtime(_))
    }

    /// Check if this is an actor error
    pub fn is_actor(&self) -> bool {
        matches!(self, Self::Actor(_))
    }
}

impl From<anyhow::Error> for PulsingError {
    fn from(err: anyhow::Error) -> Self {
        // Try to downcast to known error types
        if let Some(runtime_err) = err.downcast_ref::<RuntimeError>() {
            return Self::Runtime(runtime_err.clone());
        }
        if let Some(actor_err) = err.downcast_ref::<ActorError>() {
            return Self::Actor(actor_err.clone());
        }
        // Try to downcast to PulsingError itself
        if let Some(pulsing_err) = err.downcast_ref::<PulsingError>() {
            return pulsing_err.clone();
        }
        // Default to runtime error for unknown errors
        Self::Runtime(RuntimeError::Other(err.to_string()))
    }
}

// Implement Clone for PulsingError to support downcast
impl Clone for PulsingError {
    fn clone(&self) -> Self {
        match self {
            Self::Runtime(e) => Self::Runtime(e.clone()),
            Self::Actor(e) => Self::Actor(e.clone()),
        }
    }
}

/// Runtime errors: Framework/system-level errors
///
/// These errors occur at the framework level and are not caused by user code.
/// Examples: transport failures, cluster issues, configuration errors, etc.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    // =========================================================================
    // Actor system errors (framework-level)
    // =========================================================================
    /// Actor not found by name or ID
    #[error("Actor not found: {name}")]
    ActorNotFound { name: String },

    /// Actor already exists with the given name
    #[error("Actor already exists: {name}")]
    ActorAlreadyExists { name: String },

    /// Actor is not local to this node
    #[error("Actor is not local: {name}")]
    ActorNotLocal { name: String },

    /// Actor has stopped and cannot process messages
    #[error("Actor stopped: {name}")]
    ActorStopped { name: String },

    /// Actor mailbox is full
    #[error("Actor mailbox full: {name}")]
    ActorMailboxFull { name: String },

    /// Invalid actor path format
    #[error("Invalid actor path: {path}")]
    InvalidActorPath { path: String },

    /// Message type mismatch
    #[error("Message type mismatch: expected {expected}, got {actual}")]
    MessageTypeMismatch { expected: String, actual: String },

    /// Actor spawn failed
    #[error("Failed to spawn actor: {reason}")]
    ActorSpawnFailed { reason: String },

    // =========================================================================
    // Transport errors
    // =========================================================================
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

    // =========================================================================
    // Cluster errors
    // =========================================================================
    /// Cluster not initialized
    #[error("Cluster not initialized")]
    ClusterNotInitialized,

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

    // =========================================================================
    // Configuration errors
    // =========================================================================
    /// Invalid configuration value
    #[error("Invalid configuration: {field} = {value} ({reason})")]
    InvalidConfigValue {
        field: String,
        value: String,
        reason: String,
    },

    /// Missing required configuration
    #[error("Missing required configuration: {field}")]
    MissingRequiredConfig { field: String },

    /// Conflicting configuration options
    #[error("Conflicting configuration: {reason}")]
    ConflictingConfig { reason: String },

    /// Address parsing error
    #[error("Invalid address '{addr}': {reason}")]
    InvalidAddress { addr: String, reason: String },

    // =========================================================================
    // Other runtime errors
    // =========================================================================
    /// I/O errors
    #[error("I/O error: {0}")]
    Io(String),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Generic runtime errors
    #[error("{0}")]
    Other(String),
}

impl RuntimeError {
    // =========================================================================
    // Actor system error constructors
    // =========================================================================

    /// Create an "actor not found" error
    pub fn actor_not_found(name: impl Into<String>) -> Self {
        Self::ActorNotFound { name: name.into() }
    }

    /// Create an "actor already exists" error
    pub fn actor_already_exists(name: impl Into<String>) -> Self {
        Self::ActorAlreadyExists { name: name.into() }
    }

    /// Create an "actor not local" error
    pub fn actor_not_local(name: impl Into<String>) -> Self {
        Self::ActorNotLocal { name: name.into() }
    }

    /// Create an "actor stopped" error
    pub fn actor_stopped(name: impl Into<String>) -> Self {
        Self::ActorStopped { name: name.into() }
    }

    /// Create an "actor mailbox full" error
    pub fn actor_mailbox_full(name: impl Into<String>) -> Self {
        Self::ActorMailboxFull { name: name.into() }
    }

    /// Create an "invalid actor path" error
    pub fn invalid_actor_path(path: impl Into<String>) -> Self {
        Self::InvalidActorPath { path: path.into() }
    }

    /// Create a "message type mismatch" error
    pub fn message_type_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::MessageTypeMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create an "actor spawn failed" error
    pub fn actor_spawn_failed(reason: impl Into<String>) -> Self {
        Self::ActorSpawnFailed {
            reason: reason.into(),
        }
    }

    // =========================================================================
    // Transport error constructors
    // =========================================================================

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

    /// Create a protocol error
    pub fn protocol_error(reason: impl Into<String>) -> Self {
        Self::ProtocolError {
            reason: reason.into(),
        }
    }

    // =========================================================================
    // Cluster error constructors
    // =========================================================================

    /// Create a "cluster not initialized" error
    pub fn cluster_not_initialized() -> Self {
        Self::ClusterNotInitialized
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

    // =========================================================================
    // Config error constructors
    // =========================================================================

    /// Create an "invalid config value" error
    pub fn invalid_config_value(
        field: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidConfigValue {
            field: field.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// Create a "missing required config" error
    pub fn missing_required_config(field: impl Into<String>) -> Self {
        Self::MissingRequiredConfig {
            field: field.into(),
        }
    }

    /// Create a "conflicting config" error
    pub fn conflicting_config(reason: impl Into<String>) -> Self {
        Self::ConflictingConfig {
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

    // =========================================================================
    // Other error constructors
    // =========================================================================

    /// Create a serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }

    /// Create a generic runtime error
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Create an I/O error from std::io::Error
    pub fn io(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

impl From<std::io::Error> for RuntimeError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

/// Actor errors: User Actor execution errors
///
/// These errors are raised by user code during Actor execution.
/// They are distinct from RuntimeError which are framework-level errors.
#[derive(Error, Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ActorError {
    /// Business error: User input error, business logic error
    /// These are recoverable and should be returned to the caller
    #[error("Business error [{code}]: {message}")]
    Business {
        code: u32,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },

    /// System error: Internal error, resource error
    /// May trigger Actor restart depending on recoverable flag
    #[error("System error: {error}")]
    System { error: String, recoverable: bool },

    /// Timeout error: Operation timed out
    /// Usually recoverable, can be retried
    #[error("Timeout: operation '{operation}' timed out after {duration_ms}ms")]
    Timeout { operation: String, duration_ms: u64 },

    /// Unsupported operation
    #[error("Unsupported operation: {operation}")]
    Unsupported { operation: String },
}

impl ActorError {
    /// Create a business error
    pub fn business(code: u32, message: impl Into<String>, details: Option<String>) -> Self {
        Self::Business {
            code,
            message: message.into(),
            details,
        }
    }

    /// Create a system error
    pub fn system(error: impl Into<String>, recoverable: bool) -> Self {
        Self::System {
            error: error.into(),
            recoverable,
        }
    }

    /// Create a timeout error
    pub fn timeout(operation: impl Into<String>, duration_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            duration_ms,
        }
    }

    /// Create an unsupported operation error
    pub fn unsupported(operation: impl Into<String>) -> Self {
        Self::Unsupported {
            operation: operation.into(),
        }
    }

    /// Check if this error is recoverable
    ///
    /// - Business errors: always recoverable (return to caller)
    /// - System errors: depends on recoverable flag
    /// - Timeout errors: usually recoverable (can retry)
    /// - Unsupported errors: not recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Business { .. } => true,
            Self::System { recoverable, .. } => *recoverable,
            Self::Timeout { .. } => true,
            Self::Unsupported { .. } => false,
        }
    }

    /// Check if this is a business error
    pub fn is_business(&self) -> bool {
        matches!(self, Self::Business { .. })
    }

    /// Check if this is a system error
    pub fn is_system(&self) -> bool {
        matches!(self, Self::System { .. })
    }

    /// Check if this is a timeout error
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }
}

// =============================================================================
// Legacy type aliases for backward compatibility
// =============================================================================

/// Legacy: TransportError (now part of RuntimeError)
#[deprecated(note = "Use RuntimeError instead")]
pub type TransportError = RuntimeError;

/// Legacy: ClusterError (now part of RuntimeError)
#[deprecated(note = "Use RuntimeError instead")]
pub type ClusterError = RuntimeError;

/// Legacy: ConfigError (now part of RuntimeError)
#[deprecated(note = "Use RuntimeError instead")]
pub type ConfigError = RuntimeError;

/// Convenience type alias for results using PulsingError
pub type Result<T> = std::result::Result<T, PulsingError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_error_display() {
        let err = RuntimeError::actor_not_found("my-actor");
        assert!(err.to_string().contains("my-actor"));

        let err = RuntimeError::connection_failed("127.0.0.1:8000", "connection refused");
        assert!(err.to_string().contains("127.0.0.1:8000"));
        assert!(err.to_string().contains("refused"));
    }

    #[test]
    fn test_actor_error_display() {
        let err = ActorError::business(400, "Invalid input", None);
        assert!(err.to_string().contains("400"));
        assert!(err.to_string().contains("Invalid input"));

        let err = ActorError::system("Database error", true);
        assert!(err.to_string().contains("Database error"));
    }

    #[test]
    fn test_pulsing_error_from_runtime_error() {
        let runtime_err = RuntimeError::actor_not_found("test");
        let pulsing_err: PulsingError = runtime_err.into();

        assert!(matches!(pulsing_err, PulsingError::Runtime(_)));
        assert!(pulsing_err.to_string().contains("test"));
    }

    #[test]
    fn test_pulsing_error_from_actor_error() {
        let actor_err = ActorError::business(400, "test", None);
        let pulsing_err: PulsingError = actor_err.into();

        assert!(matches!(pulsing_err, PulsingError::Actor(_)));
        assert!(pulsing_err.to_string().contains("test"));
    }

    #[test]
    fn test_error_classification() {
        let business_err = ActorError::business(400, "test", None);
        assert!(business_err.is_recoverable());
        assert!(business_err.is_business());

        let system_err = ActorError::system("error", true);
        assert!(system_err.is_recoverable());
        assert!(system_err.is_system());

        let timeout_err = ActorError::timeout("op", 1000);
        assert!(timeout_err.is_recoverable());
        assert!(timeout_err.is_timeout());
    }

    #[test]
    fn test_error_equality() {
        let err1 = ActorError::business(400, "test", None);
        let err2 = ActorError::business(400, "test", None);
        let err3 = ActorError::business(400, "other", None);

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
