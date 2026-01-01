//! Streaming response support for HTTP/2 transport

use crate::actor::Message;
use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_util::sync::CancellationToken;

/// A frame in a streaming response
///
/// Frames are serialized as NDJSON (newline-delimited JSON) for transmission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamFrame {
    /// Sequence number (0-indexed)
    pub seq: u64,

    /// Message type identifier
    pub msg_type: String,

    /// Message data (base64 encoded binary)
    pub data: String,

    /// Whether this is the final frame
    #[serde(default)]
    pub end: bool,

    /// Error message (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl StreamFrame {
    /// Create a new data frame
    pub fn data(seq: u64, msg_type: impl Into<String>, payload: &[u8]) -> Self {
        Self {
            seq,
            msg_type: msg_type.into(),
            data: base64_encode(payload),
            end: false,
            error: None,
        }
    }

    /// Create an end frame (no data)
    pub fn end(seq: u64) -> Self {
        Self {
            seq,
            msg_type: String::new(),
            data: String::new(),
            end: true,
            error: None,
        }
    }

    /// Create an error frame
    pub fn error(seq: u64, error: impl Into<String>) -> Self {
        Self {
            seq,
            msg_type: "error".to_string(),
            data: String::new(),
            end: true,
            error: Some(error.into()),
        }
    }

    /// Decode the data payload
    pub fn decode_data(&self) -> anyhow::Result<Vec<u8>> {
        base64_decode(&self.data)
    }

    /// Check if this frame contains an error
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Create a StreamFrame from a Message::Single
    ///
    /// The msg_type from the Message is preserved in the frame.
    /// If the message has an empty msg_type, the default_msg_type is used.
    pub fn from_message(seq: u64, msg: &Message, default_msg_type: &str) -> Self {
        match msg {
            Message::Single { msg_type, data } => {
                let frame_msg_type = if msg_type.is_empty() {
                    default_msg_type.to_string()
                } else {
                    msg_type.clone()
                };
                Self::data(seq, frame_msg_type, data)
            }
            Message::Stream { .. } => {
                // Nested streams are not supported, create an error frame
                Self::error(seq, "Nested streams are not supported")
            }
        }
    }

    /// Convert this frame to a Message::Single
    ///
    /// Returns None for end frames with no data.
    /// Returns Err for error frames.
    pub fn to_message(&self) -> anyhow::Result<Option<Message>> {
        // Check for errors first
        if let Some(ref error) = self.error {
            return Err(anyhow::anyhow!("{}", error));
        }

        // Skip end frames with no data
        if self.end && self.data.is_empty() {
            return Ok(None);
        }

        // Decode and create Message
        let data = self.decode_data()?;
        Ok(Some(Message::single(&self.msg_type, data)))
    }

    /// Serialize to NDJSON line
    pub fn to_ndjson(&self) -> anyhow::Result<Bytes> {
        let json = serde_json::to_string(self)?;
        Ok(Bytes::from(format!("{}\n", json)))
    }

    /// Parse from NDJSON line
    pub fn from_ndjson(line: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(line.trim())?)
    }
}

/// Handle to a streaming response
///
/// Implements `Stream` for consuming frames. Automatically cancels the stream
/// when dropped.
pub struct StreamHandle<T> {
    inner: Pin<Box<dyn Stream<Item = anyhow::Result<T>> + Send>>,
    cancel: CancellationToken,
}

impl<T> StreamHandle<T> {
    /// Create a new stream handle
    pub fn new(
        stream: impl Stream<Item = anyhow::Result<T>> + Send + 'static,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            cancel,
        }
    }

    /// Explicitly cancel the stream
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Check if the stream has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Get a clone of the cancellation token
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

impl<T> Stream for StreamHandle<T> {
    type Item = anyhow::Result<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.cancel.is_cancelled() {
            return Poll::Ready(None);
        }
        self.inner.as_mut().poll_next(cx)
    }
}

impl<T> Drop for StreamHandle<T> {
    fn drop(&mut self) {
        // Cancel the stream when dropped
        self.cancel.cancel();
    }
}

// Base64 encoding/decoding helpers
use base64::{engine::general_purpose::STANDARD, Engine};

fn base64_encode(data: &[u8]) -> String {
    STANDARD.encode(data)
}

fn base64_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    STANDARD
        .decode(s)
        .map_err(|e| anyhow::anyhow!("Base64 decode error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_frame_data() {
        let frame = StreamFrame::data(0, "test", b"hello");
        assert_eq!(frame.seq, 0);
        assert_eq!(frame.msg_type, "test");
        assert!(!frame.end);
        assert!(frame.error.is_none());

        let decoded = frame.decode_data().unwrap();
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn test_stream_frame_end() {
        let frame = StreamFrame::end(5);
        assert_eq!(frame.seq, 5);
        assert!(frame.end);
        assert!(frame.error.is_none());
    }

    #[test]
    fn test_stream_frame_error() {
        let frame = StreamFrame::error(3, "something went wrong");
        assert_eq!(frame.seq, 3);
        assert!(frame.end);
        assert!(frame.is_error());
        assert_eq!(frame.error.unwrap(), "something went wrong");
    }

    #[test]
    fn test_ndjson_roundtrip() {
        let frame = StreamFrame::data(1, "token", b"world");
        let json = frame.to_ndjson().unwrap();
        let parsed = StreamFrame::from_ndjson(std::str::from_utf8(&json).unwrap()).unwrap();

        assert_eq!(parsed.seq, 1);
        assert_eq!(parsed.msg_type, "token");
        assert_eq!(parsed.decode_data().unwrap(), b"world");
    }

    #[test]
    fn test_base64_roundtrip() {
        let original = b"Hello, World! \x00\x01\x02\xff";
        let encoded = base64_encode(original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_from_message_with_msg_type() {
        let msg = Message::single("token", b"hello");
        let frame = StreamFrame::from_message(0, &msg, "default");

        assert_eq!(frame.seq, 0);
        assert_eq!(frame.msg_type, "token");
        assert_eq!(frame.decode_data().unwrap(), b"hello");
    }

    #[test]
    fn test_from_message_uses_default_msg_type() {
        // Empty msg_type should use default
        let msg = Message::single("", b"hello");
        let frame = StreamFrame::from_message(0, &msg, "default_type");

        assert_eq!(frame.msg_type, "default_type");
    }

    #[test]
    fn test_to_message_roundtrip() {
        let original = Message::single("chunk", b"data");
        let frame = StreamFrame::from_message(0, &original, "default");
        let recovered = frame.to_message().unwrap().unwrap();

        if let Message::Single { msg_type, data } = recovered {
            assert_eq!(msg_type, "chunk");
            assert_eq!(data, b"data");
        } else {
            panic!("Expected Single message");
        }
    }

    #[test]
    fn test_to_message_end_frame() {
        let frame = StreamFrame::end(0);
        let msg = frame.to_message().unwrap();
        assert!(msg.is_none());
    }

    #[test]
    fn test_to_message_error_frame() {
        let frame = StreamFrame::error(0, "test error");
        let result = frame.to_message();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test error"));
    }
}
