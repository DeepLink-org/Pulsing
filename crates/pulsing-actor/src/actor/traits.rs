//! Core Actor traits and types

use async_trait::async_trait;
use futures::Stream;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::pin::Pin;
use thiserror::Error;
use tokio::sync::mpsc;

// ============================================================================
// Identifiers
// ============================================================================

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
}

impl fmt::Display for ActorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.node().0, self.local_id())
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

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

// ============================================================================
// Messaging
// ============================================================================

/// Stream type for payload data
pub type PayloadStream = Pin<Box<dyn Stream<Item = anyhow::Result<Vec<u8>>> + Send>>;

/// Message stream type (for streaming scenarios)
pub type MessageStream = Pin<Box<dyn Stream<Item = anyhow::Result<Message>> + Send>>;

/// Unified message type for both requests and responses
///
/// Message is an enum with two variants:
/// - `Single`: for traditional request-response with a single data payload
/// - `Stream`: for streaming scenarios with a payload stream
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

    /// Pack a serializable value into a message
    ///
    /// Uses `std::any::type_name` to automatically generate the message type.
    pub fn pack<M: Serialize + 'static>(msg: &M) -> anyhow::Result<Self> {
        Ok(Message::Single {
            msg_type: std::any::type_name::<M>().to_string(),
            data: bincode::serialize(msg)?,
        })
    }

    /// Unpack (deserialize) the message data into a specific type
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

// ============================================================================
// Actor Trait
// ============================================================================

/// Actor context passed to handlers
pub use super::context::ActorContext;

/// Core Actor trait
///
/// Implement this trait to create an actor.
#[async_trait]
pub trait Actor: Send + Sync + 'static {
    /// Get actor metadata for diagnostics (optional).
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Called when the actor starts.
    async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// Called when the actor stops.
    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// Handle a message and produce a response.
    ///
    /// This is the unified handler for all message patterns (RPC, Streaming).
    ///
    /// # Patterns
    ///
    /// 1. **Single Request -> Single Response** (Standard RPC)
    /// ```ignore
    /// async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
    ///     let req: MyRequest = msg.unpack()?;
    ///     Message::pack(&MyResponse { .. })
    /// }
    /// ```
    ///
    /// 2. **Single Request -> Stream Response** (Server Streaming)
    /// ```ignore
    /// async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
    ///     let (tx, rx) = mpsc::channel(32);
    ///     tokio::spawn(async move {
    ///         for i in 0..10 {
    ///             tx.send(Ok(bincode::serialize(&i).unwrap())).await;
    ///         }
    ///     });
    ///     Ok(Message::from_channel("StreamResponse", rx))
    /// }
    /// ```
    ///
    /// 3. **Stream Request -> Single Response** (Client Streaming)
    /// ```ignore
    /// async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
    ///     let mut stream = match msg {
    ///         Message::Stream { stream, .. } => stream,
    ///         _ => return Err(anyhow::anyhow!("Expected stream")),
    ///     };
    ///     let mut sum = 0;
    ///     while let Some(chunk) = stream.next().await {
    ///         let val: i32 = bincode::deserialize(&chunk?)?;
    ///         sum += val;
    ///     }
    ///     Message::pack(&sum)
    /// }
    /// ```
    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> anyhow::Result<Message> {
        Err(anyhow::anyhow!(
            "Actor {} does not handle message type: {}",
            ctx.id(),
            msg.msg_type()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestMessage {
        value: i32,
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

        assert!(message.msg_type().ends_with("TestMessage"));
        assert!(message.is_single());

        let decoded: TestMessage = message.unpack().unwrap();
        assert_eq!(decoded.value, 42);
    }

    #[test]
    fn test_message_response() {
        let response = Message::single("", b"hello");
        assert!(response.msg_type().is_empty());
        assert!(response.is_single());

        let Message::Single { data, .. } = response else {
            panic!("expected single")
        };
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_message_request() {
        let request = Message::single("Echo", b"hello");
        assert!(!request.msg_type().is_empty());
        assert_eq!(request.msg_type(), "Echo");
    }

    #[tokio::test]
    async fn test_message_server_streaming() {
        use tokio_stream::StreamExt;

        // Simulate a server streaming response
        let (tx, rx) = mpsc::channel(10);
        let msg = Message::from_channel("StreamResponse", rx);

        assert!(msg.is_stream());

        tokio::spawn(async move {
            tx.send(Ok(vec![1])).await.unwrap();
            tx.send(Ok(vec![2])).await.unwrap();
            tx.send(Ok(vec![3])).await.unwrap();
        });

        let Message::Stream { mut stream, .. } = msg else {
            panic!("expected stream")
        };

        let mut values = Vec::new();
        while let Some(item) = stream.next().await {
            values.push(item.unwrap()[0]);
        }

        assert_eq!(values, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_message_client_streaming() {
        use tokio_stream::StreamExt;

        // Simulate a client streaming request
        let (tx, rx) = mpsc::channel(10);
        let msg = Message::from_channel("StreamRequest", rx);

        tokio::spawn(async move {
            tx.send(Ok(vec![10])).await.unwrap();
            tx.send(Ok(vec![20])).await.unwrap();
        });

        let Message::Stream { mut stream, .. } = msg else {
            panic!("expected stream")
        };

        let mut sum = 0;
        while let Some(item) = stream.next().await {
            sum += item.unwrap()[0];
        }

        assert_eq!(sum, 30);
    }
}
