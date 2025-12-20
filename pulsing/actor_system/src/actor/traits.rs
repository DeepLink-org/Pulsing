//! Core Actor traits and types

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// Reason why an actor stopped
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StopReason {
    /// Normal shutdown (graceful stop)
    Normal,
    /// Actor panicked or encountered an unrecoverable error
    Failed(String),
    /// Actor was killed/aborted
    Killed,
    /// System is shutting down
    SystemShutdown,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::Normal => write!(f, "Normal"),
            StopReason::Failed(msg) => write!(f, "Failed: {}", msg),
            StopReason::Killed => write!(f, "Killed"),
            StopReason::SystemShutdown => write!(f, "SystemShutdown"),
        }
    }
}

/// Message sent to watchers when a watched actor terminates
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Terminated {
    /// The actor that terminated
    pub actor_id: ActorId,
    /// Reason for termination
    pub reason: StopReason,
}

impl Message for Terminated {
    fn type_id() -> &'static str {
        "Terminated"
    }
}

/// Node identifier in the cluster
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    /// Generate a new unique NodeId
    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Create from string
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the inner string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Actor identifier (globally unique)
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActorId {
    /// Node where the actor resides
    pub node: NodeId,
    /// Actor name (unique within the node)
    pub name: String,
}

impl ActorId {
    /// Create a new ActorId
    pub fn new(node: NodeId, name: impl Into<String>) -> Self {
        Self {
            node,
            name: name.into(),
        }
    }

    /// Create a local actor id (node will be set when spawned)
    pub fn local(name: impl Into<String>) -> Self {
        Self {
            node: NodeId::new("local"),
            name: name.into(),
        }
    }
}

impl fmt::Display for ActorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.name, self.node)
    }
}

/// Message trait - all messages must be serializable
pub trait Message: Serialize + DeserializeOwned + Send + Sync + 'static {
    /// Unique type identifier for routing
    fn type_id() -> &'static str
    where
        Self: Sized;
}

/// Raw message for type-erased handling
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawMessage {
    /// Message type identifier
    pub msg_type: String,
    /// Serialized payload
    pub payload: Vec<u8>,
}

impl RawMessage {
    /// Create a new raw message from a typed message
    pub fn from_message<M: Message>(msg: &M) -> anyhow::Result<Self> {
        Ok(Self {
            msg_type: M::type_id().to_string(),
            payload: bincode::serialize(msg)?,
        })
    }

    /// Deserialize into a typed message
    pub fn into_message<M: Message>(self) -> anyhow::Result<M> {
        Ok(bincode::deserialize(&self.payload)?)
    }

    /// Create an empty response
    pub fn empty() -> Self {
        Self {
            msg_type: "empty".to_string(),
            payload: vec![],
        }
    }
}

/// Actor context passed to handlers
pub use super::context::ActorContext;

/// Core Actor trait
#[async_trait]
pub trait Actor: Send + Sync + 'static {
    /// Get the actor's unique identifier
    fn id(&self) -> &ActorId;

    /// Get actor metadata for diagnostics.
    /// Returns a key-value map that can be used by monitoring/debugging tools.
    /// The transport layer is responsible for serialization.
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Called when the actor starts
    async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// Called when the actor stops
    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// Handle a raw message with request-response semantics (ask pattern)
    ///
    /// Override this for messages that require a response.
    /// Default implementation returns an error.
    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        Err(anyhow::anyhow!(
            "Actor {} does not handle message type: {}",
            self.id(),
            msg.msg_type
        ))
    }

    /// Handle a fire-and-forget message (tell pattern)
    ///
    /// Override this for pure tell semantics without response overhead.
    /// Default implementation calls `receive` and discards the result.
    ///
    /// # Example
    /// ```ignore
    /// async fn on_tell(&mut self, msg: RawMessage, ctx: &mut ActorContext) -> anyhow::Result<()> {
    ///     match msg.msg_type.as_str() {
    ///         "LogEvent" => {
    ///             let event: LogEvent = msg.into_message()?;
    ///             self.log(event);
    ///             Ok(())
    ///         }
    ///         _ => self.default_tell(msg, ctx).await
    ///     }
    /// }
    /// ```
    async fn on_tell(
        &mut self,
        msg: RawMessage,
        ctx: &mut ActorContext,
    ) -> anyhow::Result<()> {
        // Default: delegate to receive and discard result
        let _ = self.receive(msg, ctx).await?;
        Ok(())
    }
}

/// Typed message handler trait
#[async_trait]
pub trait Handler<M: Message>: Actor {
    /// Response type
    type Response: Message;

    /// Handle the message
    async fn handle(&mut self, msg: M, ctx: &mut ActorContext) -> Self::Response;
}

/// Trait for dispatching messages to actors (used by transport layer)
#[async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handle an incoming message for an actor
    async fn handle_message(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>>;
}

/// Unit type message implementation
impl Message for () {
    fn type_id() -> &'static str {
        "unit"
    }
}

/// Simple string message
impl Message for String {
    fn type_id() -> &'static str {
        "string"
    }
}

/// Bytes message
impl Message for Vec<u8> {
    fn type_id() -> &'static str {
        "bytes"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestMessage {
        value: i32,
    }

    impl Message for TestMessage {
        fn type_id() -> &'static str {
            "TestMessage"
        }
    }

    #[test]
    fn test_raw_message_roundtrip() {
        let msg = TestMessage { value: 42 };
        let raw = RawMessage::from_message(&msg).unwrap();
        assert_eq!(raw.msg_type, "TestMessage");

        let decoded: TestMessage = raw.into_message().unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_actor_id() {
        let node = NodeId::generate();
        let id = ActorId::new(node.clone(), "test-actor");
        assert_eq!(id.name, "test-actor");
        assert_eq!(id.node, node);
    }
}
