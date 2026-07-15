//! Core actor traits and types.

use crate::error::{PulsingError, Result, RuntimeError};
use async_trait::async_trait;
use bytes::Bytes;
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
    pub fn parse<T: DeserializeOwned>(&self, data: &[u8]) -> Result<T> {
        let to_err = |e: &(dyn std::error::Error + '_)| {
            PulsingError::from(RuntimeError::Serialization(e.to_string()))
        };
        match self {
            Format::Bincode => bincode::deserialize(data).map_err(|e| to_err(&e)),
            Format::Json => serde_json::from_slice(data).map_err(|e| to_err(&e)),
            Format::Auto => match serde_json::from_slice(data) {
                Ok(value) => Ok(value),
                Err(_) => bincode::deserialize(data).map_err(|e| to_err(&e)),
            },
        }
    }

    /// Serialize data using this format
    #[allow(dead_code)]
    pub fn serialize<T: Serialize>(&self, value: &T) -> Result<Vec<u8>> {
        let to_err = |e: &(dyn std::error::Error + '_)| {
            PulsingError::from(RuntimeError::Serialization(e.to_string()))
        };
        match self {
            Format::Bincode => bincode::serialize(value).map_err(|e| to_err(&e)),
            Format::Json => serde_json::to_vec(value).map_err(|e| to_err(&e)),
            Format::Auto => bincode::serialize(value).map_err(|e| to_err(&e)),
        }
    }
}

/// Message stream type (stream of Single messages).
pub type MessageStream = Pin<Box<dyn Stream<Item = Result<Message>> + Send>>;

/// Stable message type used when a [`TensorMessage`] crosses a remote transport.
pub const TENSOR_MESSAGE_TYPE: &str = "__pulsing_tensor_message__";

const TENSOR_WIRE_MAGIC: &[u8; 4] = b"PTM1";
const TENSOR_WIRE_FIXED_HEADER: usize = 4 + 4 + 8 + 4;
const DEFAULT_MAX_TENSOR_WIRE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const DEFAULT_MAX_TENSOR_METADATA_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_TENSOR_BUFFERS: usize = 65_536;

fn tensor_env_limit(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Maximum compatibility-wire body accepted before HTTP body aggregation.
pub fn max_tensor_wire_bytes() -> usize {
    let platform_default = usize::try_from(DEFAULT_MAX_TENSOR_WIRE_BYTES).unwrap_or(usize::MAX);
    tensor_env_limit("PULSING_MAX_TENSOR_WIRE_BYTES", platform_default)
}

pub fn max_tensor_metadata_bytes() -> usize {
    tensor_env_limit(
        "PULSING_MAX_TENSOR_METADATA_BYTES",
        DEFAULT_MAX_TENSOR_METADATA_BYTES,
    )
}

pub fn max_tensor_buffers() -> usize {
    tensor_env_limit("PULSING_MAX_TENSOR_BUFFERS", DEFAULT_MAX_TENSOR_BUFFERS)
}

/// Transport-neutral tensor payload.
///
/// Pulsing intentionally treats `metadata` as opaque bytes. Tensor schemas,
/// dtypes and shapes belong to the caller (for example PulsingQueue), while
/// Pulsing only owns routing and buffer transport. `Bytes` lets an in-process
/// message retain borrowed Python buffer owners without copying the payload.
#[derive(Clone)]
pub struct TensorMessage {
    pub version: u32,
    pub metadata: Bytes,
    pub buffers: Vec<Bytes>,
    origin: TensorBufferOrigin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorBufferOrigin {
    Borrowed,
    PackedWire,
    OwnedReceive,
}

impl TensorMessage {
    pub fn new(version: u32, metadata: impl Into<Bytes>, buffers: Vec<Bytes>) -> Self {
        Self {
            version,
            metadata: metadata.into(),
            buffers,
            origin: TensorBufferOrigin::Borrowed,
        }
    }

    /// Build a message from raw TCP receive allocations. `Vec -> Bytes` moves
    /// ownership without copying; Python then exposes these as writable final
    /// buffers retained by the resulting Tensor objects.
    pub fn from_owned_receive(
        version: u32,
        metadata: Vec<u8>,
        buffers: Vec<Vec<u8>>,
    ) -> Result<Self> {
        if metadata.len() > max_tensor_metadata_bytes() {
            return Err(PulsingError::from(RuntimeError::Serialization(
                "TensorMessage metadata exceeds configured maximum".into(),
            )));
        }
        if buffers.len() > max_tensor_buffers() {
            return Err(PulsingError::from(RuntimeError::Serialization(
                "TensorMessage buffer count exceeds configured maximum".into(),
            )));
        }
        let total = buffers.iter().try_fold(metadata.len(), |total, buffer| {
            total.checked_add(buffer.len())
        });
        // Keep this explicit match for the Rust 1.75 MSRV; Option::is_none_or
        // is newer than the version declared by the project.
        let exceeds_limit = match total {
            Some(total) => total > max_tensor_wire_bytes(),
            None => true,
        };
        if exceeds_limit {
            return Err(PulsingError::from(RuntimeError::Serialization(
                "TensorMessage payload exceeds configured maximum".into(),
            )));
        }
        Ok(Self {
            version,
            metadata: Bytes::from(metadata),
            buffers: buffers.into_iter().map(Bytes::from).collect(),
            origin: TensorBufferOrigin::OwnedReceive,
        })
    }

    pub fn requires_owned_receive_copy(&self) -> bool {
        self.origin == TensorBufferOrigin::PackedWire
    }

    pub fn owns_receive_buffers(&self) -> bool {
        self.origin == TensorBufferOrigin::OwnedReceive
    }

    /// Sum of tensor payload bytes (metadata is deliberately excluded).
    pub fn total_bytes(&self) -> usize {
        self.buffers.iter().map(Bytes::len).sum()
    }

    /// Encode the current HTTP/2 compatibility representation.
    ///
    /// This performs one payload packing copy. The dedicated tensor data plane
    /// can replace this codec later without changing the public Message API.
    pub fn encode_wire(&self) -> Result<Vec<u8>> {
        if self.buffers.len() > max_tensor_buffers() {
            return Err(PulsingError::from(RuntimeError::Serialization(format!(
                "TensorMessage buffer count {} exceeds configured maximum {}",
                self.buffers.len(),
                max_tensor_buffers()
            ))));
        }
        if self.metadata.len() > max_tensor_metadata_bytes() {
            return Err(PulsingError::from(RuntimeError::Serialization(format!(
                "TensorMessage metadata size {} exceeds configured maximum {}",
                self.metadata.len(),
                max_tensor_metadata_bytes()
            ))));
        }
        let buffer_count = u32::try_from(self.buffers.len()).map_err(|_| {
            PulsingError::from(RuntimeError::Serialization(
                "TensorMessage has too many buffers".into(),
            ))
        })?;
        let metadata_len = u64::try_from(self.metadata.len()).map_err(|_| {
            PulsingError::from(RuntimeError::Serialization(
                "TensorMessage metadata is too large".into(),
            ))
        })?;

        let lengths_bytes = self.buffers.len().checked_mul(8).ok_or_else(|| {
            PulsingError::from(RuntimeError::Serialization(
                "TensorMessage header size overflow".into(),
            ))
        })?;
        let total_len = TENSOR_WIRE_FIXED_HEADER
            .checked_add(lengths_bytes)
            .and_then(|n| n.checked_add(self.metadata.len()))
            .and_then(|n| {
                self.buffers
                    .iter()
                    .try_fold(n, |acc, buffer| acc.checked_add(buffer.len()))
            })
            .ok_or_else(|| {
                PulsingError::from(RuntimeError::Serialization(
                    "TensorMessage payload size overflow".into(),
                ))
            })?;

        if total_len > max_tensor_wire_bytes() {
            return Err(PulsingError::from(RuntimeError::Serialization(format!(
                "TensorMessage wire size {total_len} exceeds configured maximum {}",
                max_tensor_wire_bytes()
            ))));
        }
        let mut wire = Vec::with_capacity(total_len);
        wire.extend_from_slice(TENSOR_WIRE_MAGIC);
        wire.extend_from_slice(&self.version.to_le_bytes());
        wire.extend_from_slice(&metadata_len.to_le_bytes());
        wire.extend_from_slice(&buffer_count.to_le_bytes());
        for buffer in &self.buffers {
            let len = u64::try_from(buffer.len()).map_err(|_| {
                PulsingError::from(RuntimeError::Serialization(
                    "TensorMessage buffer is too large".into(),
                ))
            })?;
            wire.extend_from_slice(&len.to_le_bytes());
        }
        wire.extend_from_slice(&self.metadata);
        for buffer in &self.buffers {
            wire.extend_from_slice(buffer);
        }
        Ok(wire)
    }

    /// Decode the compatibility wire format without copying metadata or
    /// individual tensor buffers. All returned `Bytes` are slices of `wire`.
    pub fn decode_wire(wire: Bytes) -> Result<Self> {
        let malformed = |reason: &str| {
            PulsingError::from(RuntimeError::Serialization(format!(
                "Malformed TensorMessage: {reason}"
            )))
        };

        if wire.len() > max_tensor_wire_bytes() {
            return Err(malformed("wire size exceeds configured maximum"));
        }
        if wire.len() < TENSOR_WIRE_FIXED_HEADER {
            return Err(malformed("header is truncated"));
        }
        if &wire[..4] != TENSOR_WIRE_MAGIC {
            return Err(malformed("invalid magic"));
        }

        let version = u32::from_le_bytes(wire[4..8].try_into().expect("fixed-width slice"));
        let metadata_len_u64 =
            u64::from_le_bytes(wire[8..16].try_into().expect("fixed-width slice"));
        let metadata_len = usize::try_from(metadata_len_u64)
            .map_err(|_| malformed("metadata length exceeds this platform"))?;
        let buffer_count =
            u32::from_le_bytes(wire[16..20].try_into().expect("fixed-width slice")) as usize;
        if buffer_count > max_tensor_buffers() {
            return Err(malformed("buffer count exceeds configured maximum"));
        }
        if metadata_len > max_tensor_metadata_bytes() {
            return Err(malformed("metadata size exceeds configured maximum"));
        }
        let lengths_bytes = buffer_count
            .checked_mul(8)
            .ok_or_else(|| malformed("buffer count overflow"))?;
        let payload_start = TENSOR_WIRE_FIXED_HEADER
            .checked_add(lengths_bytes)
            .ok_or_else(|| malformed("header size overflow"))?;
        if payload_start > wire.len() {
            return Err(malformed("buffer length table is truncated"));
        }

        let mut lengths = Vec::with_capacity(buffer_count);
        let mut total_payload = metadata_len;
        for index in 0..buffer_count {
            let offset = TENSOR_WIRE_FIXED_HEADER + index * 8;
            let len_u64 = u64::from_le_bytes(
                wire[offset..offset + 8]
                    .try_into()
                    .expect("validated length table"),
            );
            let len = usize::try_from(len_u64)
                .map_err(|_| malformed("buffer length exceeds this platform"))?;
            total_payload = total_payload
                .checked_add(len)
                .ok_or_else(|| malformed("payload size overflow"))?;
            lengths.push(len);
        }

        let expected_len = payload_start
            .checked_add(total_payload)
            .ok_or_else(|| malformed("message size overflow"))?;
        if expected_len != wire.len() {
            return Err(malformed(if expected_len > wire.len() {
                "payload is truncated"
            } else {
                "payload has trailing bytes"
            }));
        }

        let metadata_end = payload_start + metadata_len;
        let metadata = wire.slice(payload_start..metadata_end);
        let mut offset = metadata_end;
        let buffers = lengths
            .into_iter()
            .map(|len| {
                let end = offset + len;
                let buffer = wire.slice(offset..end);
                offset = end;
                buffer
            })
            .collect();

        Ok(Self {
            version,
            metadata,
            buffers,
            origin: TensorBufferOrigin::PackedWire,
        })
    }
}

impl fmt::Debug for TensorMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorMessage")
            .field("version", &self.version)
            .field("metadata_len", &self.metadata.len())
            .field("buffer_count", &self.buffers.len())
            .field("total_bytes", &self.total_bytes())
            .finish()
    }
}

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
    /// Opaque tensor metadata plus one or more contiguous payload buffers.
    Tensor(TensorMessage),
}

impl Message {
    pub fn single(msg_type: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Message::Single {
            msg_type: msg_type.into(),
            data: data.into(),
        }
    }

    pub fn pack<M: Serialize + 'static>(msg: &M) -> Result<Self> {
        bincode::serialize(msg)
            .map_err(|e| PulsingError::from(RuntimeError::Serialization(e.to_string())))
            .map(|data| Message::Single {
                msg_type: std::any::type_name::<M>().to_string(),
                data,
            })
    }

    pub fn tensor(version: u32, metadata: impl Into<Bytes>, buffers: Vec<Bytes>) -> Self {
        Message::Tensor(TensorMessage::new(version, metadata, buffers))
    }

    pub fn unpack<M: DeserializeOwned>(self) -> Result<M> {
        match self {
            Message::Single { data, .. } => bincode::deserialize(&data)
                .map_err(|e| PulsingError::from(RuntimeError::Serialization(e.to_string()))),
            Message::Stream { .. } => Err(PulsingError::from(RuntimeError::Other(
                "Cannot unpack stream message".into(),
            ))),
            Message::Tensor(_) => Err(PulsingError::from(RuntimeError::Other(
                "Cannot unpack tensor message".into(),
            ))),
        }
    }

    /// Parse message data with auto-detection (JSON first, then bincode)
    pub fn parse<M: DeserializeOwned>(&self) -> Result<M> {
        match self {
            Message::Single { data, .. } => Format::Auto.parse(data),
            Message::Stream { .. } => Err(PulsingError::from(RuntimeError::Other(
                "Cannot parse stream message".into(),
            ))),
            Message::Tensor(_) => Err(PulsingError::from(RuntimeError::Other(
                "Cannot parse tensor message".into(),
            ))),
        }
    }

    pub fn from_channel(
        default_msg_type: impl Into<String>,
        rx: mpsc::Receiver<Result<Message>>,
    ) -> Self {
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Message::Stream {
            default_msg_type: default_msg_type.into(),
            stream: Box::pin(stream),
        }
    }

    pub fn stream<S>(default_msg_type: impl Into<String>, stream: S) -> Self
    where
        S: Stream<Item = Result<Message>> + Send + 'static,
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
            Message::Tensor(_) => TENSOR_MESSAGE_TYPE,
        }
    }

    pub fn is_single(&self) -> bool {
        matches!(self, Message::Single { .. })
    }

    pub fn is_stream(&self) -> bool {
        matches!(self, Message::Stream { .. })
    }

    pub fn is_tensor(&self) -> bool {
        matches!(self, Message::Tensor(_))
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
            Message::Tensor(tensor) => tensor.fmt(f),
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
    async fn on_start(&mut self, _ctx: &mut ActorContext) -> Result<()> {
        Ok(())
    }

    /// Called when the actor stops.
    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> Result<()> {
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
    async fn receive(&mut self, msg: Message, ctx: &mut ActorContext) -> Result<Message> {
        Err(PulsingError::from(RuntimeError::Other(format!(
            "Actor {} does not handle message type: {}",
            ctx.id(),
            msg.msg_type()
        ))))
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

    #[test]
    fn test_tensor_message_wire_roundtrip() {
        let original = TensorMessage::new(
            7,
            Bytes::from_static(b"opaque-schema"),
            vec![
                Bytes::from_static(b"abc"),
                Bytes::new(),
                Bytes::from_static(b"defg"),
            ],
        );
        let wire = Bytes::from(original.encode_wire().unwrap());
        let wire_start = wire.as_ptr() as usize;
        let wire_end = wire_start + wire.len();

        let decoded = TensorMessage::decode_wire(wire).unwrap();
        assert_eq!(decoded.version, 7);
        assert_eq!(decoded.metadata, Bytes::from_static(b"opaque-schema"));
        assert_eq!(decoded.buffers.len(), 3);
        assert_eq!(&decoded.buffers[0][..], b"abc");
        assert!(decoded.buffers[1].is_empty());
        assert_eq!(&decoded.buffers[2][..], b"defg");
        assert_eq!(decoded.total_bytes(), 7);

        // decode_wire returns slices of the received allocation rather than
        // copying every tensor into a fresh Vec.
        let first_ptr = decoded.buffers[0].as_ptr() as usize;
        assert!((wire_start..wire_end).contains(&first_ptr));
    }

    #[test]
    fn test_tensor_message_allows_control_only_payload() {
        let original = TensorMessage::new(1, Bytes::from_static(b"control"), Vec::new());
        let decoded =
            TensorMessage::decode_wire(Bytes::from(original.encode_wire().unwrap())).unwrap();
        assert_eq!(decoded.metadata, Bytes::from_static(b"control"));
        assert!(decoded.buffers.is_empty());
        assert_eq!(decoded.total_bytes(), 0);
    }

    #[test]
    fn test_tensor_message_rejects_truncated_wire() {
        let original = TensorMessage::new(
            1,
            Bytes::from_static(b"meta"),
            vec![Bytes::from_static(b"payload")],
        );
        let mut wire = original.encode_wire().unwrap();
        wire.pop();
        let error = TensorMessage::decode_wire(Bytes::from(wire)).unwrap_err();
        assert!(error.to_string().contains("payload is truncated"));
    }

    #[tokio::test]
    async fn test_message_server_streaming() {
        // Simulate a server streaming response with Message stream
        let (tx, rx) = mpsc::channel::<Result<Message>>(10);
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
        let (tx, rx) = mpsc::channel::<Result<Message>>(10);
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
        let (tx, rx) = mpsc::channel::<Result<Message>>(10);
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
