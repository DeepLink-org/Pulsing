//! Streaming response support for HTTP/2 transport

use crate::actor::{Message, MessageStream};
use bytes::Bytes;
use futures::{Stream, StreamExt};
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

    /// Create a final data frame
    pub fn data_end(seq: u64, msg_type: impl Into<String>, payload: &[u8]) -> Self {
        Self {
            seq,
            msg_type: msg_type.into(),
            data: base64_encode(payload),
            end: true,
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

/// Convert a MessageStream to StreamFrame stream
pub fn to_frame_stream(stream: MessageStream) -> impl Stream<Item = anyhow::Result<StreamFrame>> {
    stream.enumerate().map(|(seq, result)| {
        result.and_then(|msg| {
            let Message::Single { msg_type, data } = msg else {
                return Err(anyhow::anyhow!("Expected single message in stream"));
            };
            Ok(StreamFrame::data(seq as u64, &msg_type, &data))
        })
    })
}

/// Convert a stream of StreamFrames to NDJSON bytes
pub fn to_ndjson_stream(
    stream: impl Stream<Item = anyhow::Result<StreamFrame>>,
) -> impl Stream<Item = anyhow::Result<Bytes>> {
    stream.map(|result| result.and_then(|frame| frame.to_ndjson()))
}

/// Parse an NDJSON stream back into MessageStream
pub fn from_ndjson_stream(
    stream: impl Stream<Item = anyhow::Result<Bytes>> + Send + 'static,
) -> MessageStream {
    let parsed = stream.filter_map(|result| async move {
        match result {
            Ok(bytes) => {
                let line = match std::str::from_utf8(&bytes) {
                    Ok(s) => s,
                    Err(e) => return Some(Err(anyhow::anyhow!("Invalid UTF-8: {}", e))),
                };

                match StreamFrame::from_ndjson(line) {
                    Ok(frame) => {
                        if frame.end && frame.data.is_empty() {
                            // End frame with no data
                            None
                        } else if let Some(error) = frame.error {
                            Some(Err(anyhow::anyhow!("{}", error)))
                        } else {
                            match frame.decode_data() {
                                Ok(payload) => Some(Ok(Message::single(&frame.msg_type, payload))),
                                Err(e) => Some(Err(e)),
                            }
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            }
            Err(e) => Some(Err(e)),
        }
    });

    Box::pin(parsed)
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
}
