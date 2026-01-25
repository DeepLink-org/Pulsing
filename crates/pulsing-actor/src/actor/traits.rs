//! Core actor traits and types.

use async_trait::async_trait;
use futures::Stream;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::pin::Pin;
use thiserror::Error;
use tokio::sync::mpsc;

/// Node identifier in the cluster (0 = local).
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct NodeId(pub u128);

impl NodeId {
    pub const LOCAL: NodeId = NodeId(0);

    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().as_u128())
    }

    pub fn new(id: u128) -> Self {
        Self(id)
    }

    pub fn is_local(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for NodeId {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_local() {
            write!(f, "0")
        } else {
            // Format as UUID string for better readability
            let uuid = uuid::Uuid::from_u128(self.0);
            write!(f, "{}", uuid.simple())
        }
    }
}

/// Actor identifier (globally unique).
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct ActorId(pub u128);

impl ActorId {
    /// Generate a new unique ActorId using UUID v4
    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().as_u128())
    }

    /// Create an ActorId from a u128 value
    pub fn new(id: u128) -> Self {
        Self(id)
    }
}

impl fmt::Display for ActorId {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format as UUID string for better readability
        let uuid = uuid::Uuid::from_u128(self.0);
        write!(f, "{}", uuid.simple())
    }
}

/// Reason why an actor stopped.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
pub enum StopReason {
    #[error("Normal")]
    Normal,
    #[error("Failed: {0}")]
    Failed(String),
    #[error("Killed")]
    Killed,
    #[error("SystemShutdown")]
    SystemShutdown,
}

/// Message serialization format
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Binary format (bincode)
    Bincode,
    /// JSON format (serde_json)
    Json,
    /// Auto-detect format (try JSON first, then bincode)
    Auto,
}

impl Format {
    /// Parse data using this format
    pub fn parse<T: DeserializeOwned>(&self, data: &[u8]) -> anyhow::Result<T> {
        match self {
            Format::Bincode => Ok(bincode::deserialize(data)?),
            Format::Json => Ok(serde_json::from_slice(data)?),
            Format::Auto => {
                // Try JSON first for Python compatibility, then bincode
                match serde_json::from_slice(data) {
                    Ok(value) => Ok(value),
                    Err(_) => Ok(bincode::deserialize(data)?),
                }
            }
        }
    }

    /// Serialize data using this format
    #[allow(dead_code)]
    pub fn serialize<T: Serialize>(&self, value: &T) -> anyhow::Result<Vec<u8>> {
        match self {
            Format::Bincode => Ok(bincode::serialize(value)?),
            Format::Json => Ok(serde_json::to_vec(value)?),
            Format::Auto => {
                // Default to bincode for Auto serialization
                Ok(bincode::serialize(value)?)
            }
        }
    }
}

/// Message stream type (stream of Single messages).
pub type MessageStream = Pin<Box<dyn Stream<Item = anyhow::Result<Message>> + Send>>;

/// Unified message type for both requests and responses.
pub enum Message {
    Single {
        msg_type: String,
        data: Vec<u8>,
    },
    Stream {
        default_msg_type: String,
        stream: MessageStream,
    },
}

impl Message {
    pub fn single(msg_type: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Message::Single {
            msg_type: msg_type.into(),
            data: data.into(),
        }
    }

    pub fn pack<M: Serialize + 'static>(msg: &M) -> anyhow::Result<Self> {
        Ok(Message::Single {
            msg_type: std::any::type_name::<M>().to_string(),
            data: bincode::serialize(msg)?,
        })
    }

    pub fn unpack<M: DeserializeOwned>(self) -> anyhow::Result<M> {
        match self {
            Message::Single { data, .. } => Ok(bincode::deserialize(&data)?),
            Message::Stream { .. } => Err(anyhow::anyhow!("Cannot unpack stream message")),
        }
    }

    /// Parse message data with auto-detection (JSON first, then bincode)
    pub fn parse<M: DeserializeOwned>(&self) -> anyhow::Result<M> {
        match self {
            Message::Single { data, .. } => Format::Auto.parse(data),
            Message::Stream { .. } => Err(anyhow::anyhow!("Cannot parse stream message")),
        }
    }

    pub fn from_channel(
        default_msg_type: impl Into<String>,
        rx: mpsc::Receiver<anyhow::Result<Message>>,
    ) -> Self {
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Message::Stream {
            default_msg_type: default_msg_type.into(),
            stream: Box::pin(stream),
        }
    }

    pub fn stream<S>(default_msg_type: impl Into<String>, stream: S) -> Self
    where
        S: Stream<Item = anyhow::Result<Message>> + Send + 'static,
    {
        Message::Stream {
            default_msg_type: default_msg_type.into(),
            stream: Box::pin(stream),
        }
    }

    pub fn msg_type(&self) -> &str {
        match self {
            Message::Single { msg_type, .. } => msg_type,
            Message::Stream {
                default_msg_type, ..
            } => default_msg_type,
        }
    }

    pub fn is_single(&self) -> bool {
        matches!(self, Message::Single { .. })
    }

    pub fn is_stream(&self) -> bool {
        matches!(self, Message::Stream { .. })
    }
}

impl fmt::Debug for Message {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Message::Single { msg_type, data } => f
                .debug_struct("Message::Single")
                .field("msg_type", msg_type)
                .field("data_len", &data.len())
                .finish(),
            Message::Stream {
                default_msg_type, ..
            } => f
                .debug_struct("Message::Stream")
                .field("default_msg_type", default_msg_type)
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
    ///             let data = bincode::serialize(&i).unwrap();
    ///             tx.send(Ok(Message::single("item", data))).await;
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
    ///         let Message::Single { data, .. } = chunk? else { continue };
    ///         let val: i32 = bincode::deserialize(&data)?;
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

/// Trait for types that can be converted into an Actor
///
/// This trait enables a uniform API for spawning both regular actors
/// and behavior-based actors using the same `spawn` and `spawn_named` methods.
///
/// # Example
///
/// ```rust,ignore
/// // Regular actor - implements Actor directly
/// struct MyActor;
/// impl Actor for MyActor { ... }
/// system.spawn(MyActor).await?;
///
/// // Behavior - implements IntoActor via BehaviorWrapper
/// fn counter(init: i32) -> Behavior<i32> { ... }
/// system.spawn(counter(0)).await?;  // Automatically wrapped
/// system.spawn_named("counter", counter(0)).await?;
/// ```
pub trait IntoActor: Send + 'static {
    /// The actor type produced by this conversion
    type Actor: Actor;

    /// Convert self into an Actor
    fn into_actor(self) -> Self::Actor;
}

/// Blanket implementation: any Actor can be converted to itself
impl<A: Actor> IntoActor for A {
    type Actor = A;

    fn into_actor(self) -> Self::Actor {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestMessage {
        value: i32,
    }

    #[test]
    fn test_actor_id() {
        let id = ActorId::generate();
        // UUID-based IDs are unique and non-zero
        assert_ne!(id.0, 0);

        // Test creating from specific value
        let id2 = ActorId::new(12345);
        assert_eq!(id2.0, 12345);
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
        // Simulate a server streaming response with Message stream
        let (tx, rx) = mpsc::channel::<anyhow::Result<Message>>(10);
        let msg = Message::from_channel("StreamResponse", rx);

        assert!(msg.is_stream());

        tokio::spawn(async move {
            tx.send(Ok(Message::single("chunk", vec![1])))
                .await
                .unwrap();
            tx.send(Ok(Message::single("chunk", vec![2])))
                .await
                .unwrap();
            tx.send(Ok(Message::single("chunk", vec![3])))
                .await
                .unwrap();
        });

        let Message::Stream { mut stream, .. } = msg else {
            panic!("expected stream")
        };

        let mut values = Vec::new();
        while let Some(item) = StreamExt::next(&mut stream).await {
            let msg: Message = item.unwrap();
            if let Message::Single { data, .. } = msg {
                values.push(data[0]);
            }
        }

        assert_eq!(values, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_message_client_streaming() {
        // Simulate a client streaming request
        let (tx, rx) = mpsc::channel::<anyhow::Result<Message>>(10);
        let msg = Message::from_channel("StreamRequest", rx);

        tokio::spawn(async move {
            tx.send(Ok(Message::single("", vec![10]))).await.unwrap();
            tx.send(Ok(Message::single("", vec![20]))).await.unwrap();
        });

        let Message::Stream { mut stream, .. } = msg else {
            panic!("expected stream")
        };

        let mut sum = 0;
        while let Some(item) = StreamExt::next(&mut stream).await {
            let msg: Message = item.unwrap();
            if let Message::Single { data, .. } = msg {
                sum += data[0];
            }
        }

        assert_eq!(sum, 30);
    }

    #[tokio::test]
    async fn test_message_stream_heterogeneous() {
        // Test heterogeneous stream - different message types in one stream
        let (tx, rx) = mpsc::channel::<anyhow::Result<Message>>(10);
        let msg = Message::from_channel("MixedStream", rx);

        tokio::spawn(async move {
            tx.send(Ok(Message::single("token", b"Hello".to_vec())))
                .await
                .unwrap();
            tx.send(Ok(Message::single("token", b" World".to_vec())))
                .await
                .unwrap();
            // Different type at the end
            tx.send(Ok(Message::single(
                "usage",
                serde_json::to_vec(&serde_json::json!({"tokens": 2})).unwrap(),
            )))
            .await
            .unwrap();
        });

        let Message::Stream { mut stream, .. } = msg else {
            panic!("expected stream")
        };

        let mut types = Vec::new();
        while let Some(item) = StreamExt::next(&mut stream).await {
            let msg: Message = item.unwrap();
            if let Message::Single { msg_type, .. } = msg {
                types.push(msg_type);
            }
        }

        assert_eq!(types, vec!["token", "token", "usage"]);
    }
}
