//! Message codec for TCP transport

use crate::actor::ActorId;
use bytes::{Buf, BufMut, BytesMut};
use serde::{Deserialize, Serialize};
use tokio_util::codec::{Decoder, Encoder};

/// Transport layer message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportMessage {
    /// Request with expected response
    Request {
        /// Request ID for correlation
        id: u64,
        /// Target actor
        actor_id: ActorId,
        /// Message type identifier
        msg_type: String,
        /// Serialized payload
        payload: Vec<u8>,
    },

    /// Response to a request
    Response {
        /// Correlation ID (matches request id)
        id: u64,
        /// Result (Ok payload or error message)
        result: Result<Vec<u8>, String>,
    },

    /// One-way message (no response expected)
    OneWay {
        /// Target actor
        actor_id: ActorId,
        /// Message type identifier
        msg_type: String,
        /// Serialized payload
        payload: Vec<u8>,
    },

    /// Ping for connection health check
    Ping { seq: u64 },

    /// Pong response
    Pong { seq: u64 },
}

impl TransportMessage {
    /// Create a request message
    pub fn request(actor_id: ActorId, msg_type: String, payload: Vec<u8>) -> (u64, Self) {
        let id = rand::random();
        (
            id,
            Self::Request {
                id,
                actor_id,
                msg_type,
                payload,
            },
        )
    }

    /// Create a response message
    pub fn response(id: u64, result: Result<Vec<u8>, String>) -> Self {
        Self::Response { id, result }
    }

    /// Create a one-way message
    pub fn one_way(actor_id: ActorId, msg_type: String, payload: Vec<u8>) -> Self {
        Self::OneWay {
            actor_id,
            msg_type,
            payload,
        }
    }

    /// Get request ID if this is a request
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Self::Request { id, .. } => Some(*id),
            _ => None,
        }
    }

    /// Get response ID if this is a response
    pub fn response_id(&self) -> Option<u64> {
        match self {
            Self::Response { id, .. } => Some(*id),
            _ => None,
        }
    }
}

/// Length-prefixed message codec
///
/// Format: [length: u32][payload: bytes]
pub struct MessageCodec {
    /// Maximum message size (to prevent OOM)
    max_size: usize,
}

impl MessageCodec {
    /// Default max message size (16 MB)
    pub const DEFAULT_MAX_SIZE: usize = 16 * 1024 * 1024;

    /// Create a new codec with default settings
    pub fn new() -> Self {
        Self {
            max_size: Self::DEFAULT_MAX_SIZE,
        }
    }

    /// Create a codec with custom max size
    pub fn with_max_size(max_size: usize) -> Self {
        Self { max_size }
    }
}

impl Default for MessageCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for MessageCodec {
    type Item = TransportMessage;
    type Error = std::io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        // Need at least 4 bytes for length
        if src.len() < 4 {
            return Ok(None);
        }

        // Read length (peek, don't consume yet)
        let len = u32::from_be_bytes([src[0], src[1], src[2], src[3]]) as usize;

        // Validate length
        if len > self.max_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Message too large: {} > {}", len, self.max_size),
            ));
        }

        // Check if we have the full message
        if src.len() < 4 + len {
            // Reserve space for the full message
            src.reserve(4 + len - src.len());
            return Ok(None);
        }

        // Consume the length prefix
        src.advance(4);

        // Read and decode the message
        let data = src.split_to(len);
        let msg: TransportMessage = bincode::deserialize(&data).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Decode error: {}", e),
            )
        })?;

        Ok(Some(msg))
    }
}

impl Encoder<TransportMessage> for MessageCodec {
    type Error = std::io::Error;

    fn encode(&mut self, item: TransportMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        let data = bincode::serialize(&item).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Encode error: {}", e),
            )
        })?;

        if data.len() > self.max_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Message too large: {} > {}", data.len(), self.max_size),
            ));
        }

        // Write length prefix + data
        dst.reserve(4 + data.len());
        dst.put_u32(data.len() as u32);
        dst.extend_from_slice(&data);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::NodeId;

    #[test]
    fn test_codec_roundtrip() {
        let mut codec = MessageCodec::new();
        let mut buf = BytesMut::new();

        let actor_id = ActorId::new(NodeId::generate(), "test");
        let (id, msg) = TransportMessage::request(actor_id, "TestMsg".to_string(), vec![1, 2, 3]);

        // Encode
        codec.encode(msg.clone(), &mut buf).unwrap();

        // Decode
        let decoded = codec.decode(&mut buf).unwrap().unwrap();

        match decoded {
            TransportMessage::Request {
                id: decoded_id,
                actor_id: _,
                msg_type,
                payload,
            } => {
                assert_eq!(decoded_id, id);
                assert_eq!(msg_type, "TestMsg");
                assert_eq!(payload, vec![1, 2, 3]);
            }
            _ => panic!("Expected Request"),
        }
    }

    #[test]
    fn test_codec_partial_read() {
        let mut codec = MessageCodec::new();
        let mut buf = BytesMut::new();

        let actor_id = ActorId::new(NodeId::generate(), "test");
        let (_, msg) = TransportMessage::request(actor_id, "TestMsg".to_string(), vec![1, 2, 3]);

        // Encode
        codec.encode(msg, &mut buf).unwrap();

        // Split into partial chunks
        let full = buf.split();
        let mut partial = BytesMut::new();
        partial.extend_from_slice(&full[..2]);

        // Should return None (incomplete)
        assert!(codec.decode(&mut partial).unwrap().is_none());

        // Add rest of the data
        partial.extend_from_slice(&full[2..]);

        // Now should decode successfully
        assert!(codec.decode(&mut partial).unwrap().is_some());
    }
}
