//! Core Actor traits and types

use async_trait::async_trait;
use futures::Stream;
use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::pin::Pin;
use thiserror::Error;
use tokio::sync::mpsc;

/// Reason why an actor stopped
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
pub enum StopReason {
    /// Normal shutdown (graceful stop)
    #[error("Normal")]
    Normal,
    /// Actor panicked or encountered an unrecoverable error
    #[error("Failed: {0}")]
    Failed(String),
    /// Actor was killed/aborted
    #[error("Killed")]
    Killed,
    /// System is shutting down
    #[error("SystemShutdown")]
    SystemShutdown,
}

/// Node identifier in the cluster (0 = local)
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Local node id (0)
    pub const LOCAL: NodeId = NodeId(0);

    /// Generate a new unique NodeId using UUID
    pub fn generate() -> Self {
        let uuid = uuid::Uuid::new_v4();
        // Use the lower 64 bits of UUID, ensure non-zero (0 is reserved for LOCAL)
        let id = uuid.as_u128() as u64;
        Self(if id == 0 { 1 } else { id })
    }

    /// Create from u64
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Check if this is the local node
    pub fn is_local(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Actor identifier (globally unique)
/// High 64 bits = node id, Low 64 bits = local actor id
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct ActorId(pub u128);

impl ActorId {
    /// Create a new ActorId from node id and local id
    pub fn new(node: NodeId, local_id: u64) -> Self {
        Self(((node.0 as u128) << 64) | (local_id as u128))
    }

    /// Create a local actor id
    pub fn local(local_id: u64) -> Self {
        Self::new(NodeId::LOCAL, local_id)
    }

    /// Get the node id
    pub fn node(&self) -> NodeId {
        NodeId((self.0 >> 64) as u64)
    }

    /// Get the local actor id
    pub fn local_id(&self) -> u64 {
        self.0 as u64
    }

    /// Check if this is a local actor
    pub fn is_local(&self) -> bool {
        self.node().is_local()
    }
}

impl fmt::Display for ActorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.node().0, self.local_id())
    }
}

// ============================================================================
// Message - unified message type supporting single and streaming
// ============================================================================

/// Stream type for payload data
pub type PayloadStream = Pin<Box<dyn Stream<Item = anyhow::Result<Vec<u8>>> + Send>>;

/// Unified message type for both requests and responses
///
/// Message is an enum with two variants:
/// - `Single`: for traditional request-response with a single data payload
/// - `Stream`: for streaming scenarios with a payload stream
///
/// # Example
/// ```ignore
/// // Create a single message with raw bytes
/// let request = Message::single("Echo", b"hello");
///
/// // Pack a serializable struct (uses type_name as msg_type)
/// let request = Message::pack(&Ping { value: 42 })?;
///
/// // Create a streaming response
/// let (tx, rx) = mpsc::channel(32);
/// let response = Message::from_channel("", rx);
///
/// // Unpack a message to a specific type
/// let ping: Ping = msg.unpack()?;
/// ```
pub enum Message {
    /// Single data message
    Single {
        /// Message type identifier (empty for responses)
        msg_type: String,
        /// Message data
        data: Vec<u8>,
    },
    /// Streaming data message
    Stream {
        /// Message type identifier (empty for responses)
        msg_type: String,
        /// Payload stream
        stream: PayloadStream,
    },
}

impl Message {
    /// Create a single message with type and data
    pub fn single(msg_type: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Message::Single {
            msg_type: msg_type.into(),
            data: data.into(),
        }
    }

    /// Create an empty single message (for responses)
    pub fn empty() -> Self {
        Message::Single {
            msg_type: String::new(),
            data: Vec::new(),
        }
    }

    /// Pack a serializable value into a message
    ///
    /// Uses `std::any::type_name` to automatically generate the message type.
    /// The type name includes the full module path (e.g., "my_crate::Ping").
    ///
    /// # Example
    /// ```ignore
    /// let msg = Message::pack(&Ping { value: 42 })?;
    /// assert!(msg.msg_type().ends_with("::Ping"));
    /// ```
    pub fn pack<M: Serialize + 'static>(msg: &M) -> anyhow::Result<Self> {
        Ok(Message::Single {
            msg_type: std::any::type_name::<M>().to_string(),
            data: bincode::serialize(msg)?,
        })
    }

    /// Unpack (deserialize) the message data into a specific type
    ///
    /// Only works for `Single` variant. Does not check the message type -
    /// the caller is responsible for matching the correct type.
    ///
    /// # Example
    /// ```ignore
    /// let ping: Ping = msg.unpack()?;
    /// ```
    pub fn unpack<M: DeserializeOwned>(self) -> anyhow::Result<M> {
        match self {
            Message::Single { data, .. } => Ok(bincode::deserialize(&data)?),
            Message::Stream { .. } => Err(anyhow::anyhow!("Cannot unpack stream message")),
        }
    }

    /// Create a streaming message from a channel receiver
    pub fn from_channel(
        msg_type: impl Into<String>,
        rx: mpsc::Receiver<anyhow::Result<Vec<u8>>>,
    ) -> Self {
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Message::Stream {
            msg_type: msg_type.into(),
            stream: Box::pin(stream),
        }
    }

    /// Create a streaming message from a stream
    pub fn stream<S>(msg_type: impl Into<String>, stream: S) -> Self
    where
        S: Stream<Item = anyhow::Result<Vec<u8>>> + Send + 'static,
    {
        Message::Stream {
            msg_type: msg_type.into(),
            stream: Box::pin(stream),
        }
    }

    /// Get message type (works for both variants)
    pub fn msg_type(&self) -> &str {
        match self {
            Message::Single { msg_type, .. } => msg_type,
            Message::Stream { msg_type, .. } => msg_type,
        }
    }

    /// Check if this is a single message
    pub fn is_single(&self) -> bool {
        matches!(self, Message::Single { .. })
    }

    /// Check if this is a streaming message
    pub fn is_stream(&self) -> bool {
        matches!(self, Message::Stream { .. })
    }

    /// Check if this message has a type
    pub fn has_type(&self) -> bool {
        !self.msg_type().is_empty()
    }
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Message::Single { msg_type, data } => f
                .debug_struct("Message::Single")
                .field("msg_type", msg_type)
                .field("data_len", &data.len())
                .finish(),
            Message::Stream { msg_type, .. } => f
                .debug_struct("Message::Stream")
                .field("msg_type", msg_type)
                .finish_non_exhaustive(),
        }
    }
}

impl Default for Message {
    fn default() -> Self {
        Self::empty()
    }
}

// Custom Serialize/Deserialize for Message (only supports Single variant)
impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Message::Single { msg_type, data } => (msg_type, data).serialize(serializer),
            Message::Stream { .. } => Err(serde::ser::Error::custom(
                "Cannot serialize streaming message",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (msg_type, data): (String, Vec<u8>) = Deserialize::deserialize(deserializer)?;
        Ok(Message::Single { msg_type, data })
    }
}

impl Clone for Message {
    fn clone(&self) -> Self {
        match self {
            Message::Single { msg_type, data } => Message::Single {
                msg_type: msg_type.clone(),
                data: data.clone(),
            },
            Message::Stream { .. } => panic!("Cannot clone streaming message"),
        }
    }
}

/// Message stream type (for streaming scenarios)
pub type MessageStream = Pin<Box<dyn Stream<Item = anyhow::Result<Message>> + Send>>;

/// Create an empty message stream
pub fn empty_stream() -> MessageStream {
    Box::pin(futures::stream::empty())
}

// ============================================================================
// Actor trait - unified interface
// ============================================================================

/// Actor context passed to handlers
pub use super::context::ActorContext;

/// Core Actor trait
///
/// Implement this trait to create an actor. The `receive` method handles all messages.
/// Use `ctx.id()` to get the actor's ID within handlers.
///
/// # Example
/// ```ignore
/// struct Counter {
///     count: i32,
/// }
///
/// #[async_trait]
/// impl Actor for Counter {
///     async fn receive(
///         &mut self,
///         msg: Message,
///         ctx: &mut ActorContext,
///     ) -> anyhow::Result<Message> {
///         if msg.msg_type().ends_with("Increment") {
///             self.count += 1;
///             return Message::pack(&self.count);
///         }
///         Err(anyhow::anyhow!("Unknown message"))
///     }
/// }
///
/// // Spawn with a name
/// let actor_ref = system.spawn("counter", Counter { count: 0 }).await?;
/// ```
#[async_trait]
pub trait Actor: Send + Sync + 'static {
    /// Get actor metadata for diagnostics (optional).
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Called when the actor starts.
    /// Use `ctx.id()` to get the actor's assigned ID.
    async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// Called when the actor stops.
    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// Handle a message and produce a response.
    ///
    /// This is the unified handler for all message patterns:
    /// - Single -> Single (traditional RPC)
    /// - Single -> Stream (server streaming, e.g., LLM generation)
    /// - Stream -> Single (client streaming)
    /// - Stream -> Stream (bidirectional streaming)
    ///
    /// For "tell" (fire-and-forget) messages, the response is ignored.
    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> anyhow::Result<Message> {
        Err(anyhow::anyhow!(
            "Actor {} does not handle message type: {}",
            ctx.id(),
            msg.msg_type()
        ))
    }
}

/// Trait for dispatching messages to actors (used by transport layer)
#[async_trait]
pub trait MessageDispatcher: Send + Sync {
    /// Handle an incoming message for an actor
    async fn dispatch(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<Message>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestMessage {
        value: i32,
    }

    #[test]
    fn test_message_serialization() {
        // Test serialization of single message
        let msg = Message::single("TestType", b"hello");
        let serialized = bincode::serialize(&msg).unwrap();
        let deserialized: Message = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.msg_type(), "TestType");
        let Message::Single { data, .. } = deserialized else {
            panic!("expected single")
        };
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_message_clone() {
        let msg = Message::single("TestType", b"hello");
        let cloned = msg.clone();
        assert_eq!(cloned.msg_type(), "TestType");
        let Message::Single { data, .. } = cloned else {
            panic!("expected single")
        };
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_actor_id() {
        let node = NodeId::generate();
        let id = ActorId::new(node, 123);
        assert_eq!(id.local_id(), 123);
        assert_eq!(id.node(), node);
    }

    #[test]
    fn test_message_single() {
        let msg = Message::single("Echo", b"hello");
        assert!(msg.is_single());
        assert!(!msg.is_stream());
        assert_eq!(msg.msg_type(), "Echo");

        let Message::Single { data, .. } = msg else {
            panic!("expected single")
        };
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_message_pack_unpack() {
        let msg = TestMessage { value: 42 };
        let message = Message::pack(&msg).unwrap();

        // type_name includes module path
        assert!(message.msg_type().ends_with("TestMessage"));
        assert!(message.is_single());

        let decoded: TestMessage = message.unpack().unwrap();
        assert_eq!(decoded.value, 42);
    }

    #[test]
    fn test_message_response() {
        // Response without type (empty string)
        let response = Message::single("", b"hello");
        assert!(!response.has_type());
        assert!(response.is_single());

        let Message::Single { data, .. } = response else {
            panic!("expected single")
        };
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_message_request() {
        // Request with type
        let request = Message::single("Echo", b"hello");
        assert!(request.has_type());
        assert_eq!(request.msg_type(), "Echo");
    }
}
