//! Streaming support for HTTP/2 transport.

use crate::actor::Message;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_util::sync::CancellationToken;

/// Frame flags.
pub const FLAG_END: u8 = 0x01;
pub const FLAG_ERROR: u8 = 0x02;

/// A frame in a streaming message.
#[derive(Debug, Clone)]
pub struct StreamFrame {
    pub msg_type: String,

    pub data: Vec<u8>,

    pub end: bool,

    pub error: Option<String>,
}

impl StreamFrame {
    pub fn data(msg_type: impl Into<String>, payload: &[u8]) -> Self {
        Self {
            msg_type: msg_type.into(),
            data: payload.to_vec(),
            end: false,
            error: None,
        }
    }

    pub fn end() -> Self {
        Self {
            msg_type: String::new(),
            data: Vec::new(),
            end: true,
            error: None,
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        Self {
            msg_type: String::new(),
            data: Vec::new(),
            end: true,
            error: Some(error.into()),
        }
    }

    #[inline]
    pub fn get_data(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    pub fn from_message(msg: &Message, default_msg_type: &str) -> Self {
        match msg {
            Message::Single { msg_type, data } => {
                let frame_msg_type = if msg_type.is_empty() {
                    default_msg_type
                } else {
                    msg_type
                };
                Self::data(frame_msg_type, data)
            }
            Message::Stream { .. } => Self::error("Nested streams are not supported"),
        }
    }

    pub fn to_message(&self) -> anyhow::Result<Option<Message>> {
        if let Some(ref error) = self.error {
            return Err(anyhow::anyhow!("{}", error));
        }
        if self.end && self.data.is_empty() {
            return Ok(None);
        }
        Ok(Some(Message::single(&self.msg_type, self.data.clone())))
    }

    /// Serialize to binary format.
    pub fn to_binary(&self) -> Bytes {
        let msg_type_bytes = self.msg_type.as_bytes();
        let error_bytes = self.error.as_ref().map(|e| e.as_bytes());

        let mut content_len = 1 + 2 + msg_type_bytes.len() + 4 + self.data.len();
        if let Some(err) = &error_bytes {
            content_len += 2 + err.len();
        }

        let mut buf = BytesMut::with_capacity(4 + content_len);

        buf.put_u32(content_len as u32);

        let mut flags = 0u8;
        if self.end {
            flags |= FLAG_END;
        }
        if self.error.is_some() {
            flags |= FLAG_ERROR;
        }
        buf.put_u8(flags);

        buf.put_u16(msg_type_bytes.len() as u16);
        buf.put_slice(msg_type_bytes);

        buf.put_u32(self.data.len() as u32);
        buf.put_slice(&self.data);

        if let Some(err) = error_bytes {
            buf.put_u16(err.len() as u16);
            buf.put_slice(err);
        }

        buf.freeze()
    }

    pub fn from_binary(mut buf: &[u8]) -> anyhow::Result<Self> {
        if buf.remaining() < 4 {
            return Err(anyhow::anyhow!("Buffer too short for length"));
        }

        let total_len = buf.get_u32() as usize;
        if buf.remaining() < total_len {
            return Err(anyhow::anyhow!(
                "Incomplete frame: expected {} bytes",
                total_len
            ));
        }

        let flags = buf.get_u8();
        let end = (flags & FLAG_END) != 0;
        let has_error = (flags & FLAG_ERROR) != 0;

        let msg_type_len = buf.get_u16() as usize;
        if buf.remaining() < msg_type_len {
            return Err(anyhow::anyhow!("Invalid msg_type length"));
        }
        let msg_type = String::from_utf8(buf[..msg_type_len].to_vec())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in msg_type: {}", e))?;
        buf.advance(msg_type_len);

        if buf.remaining() < 4 {
            return Err(anyhow::anyhow!("Missing data length"));
        }
        let data_len = buf.get_u32() as usize;
        if buf.remaining() < data_len {
            return Err(anyhow::anyhow!("Invalid data length"));
        }
        let data = buf[..data_len].to_vec();
        buf.advance(data_len);

        // Error (if flagged)
        let error = if has_error && buf.remaining() >= 2 {
            let error_len = buf.get_u16() as usize;
            if buf.remaining() >= error_len {
                Some(
                    String::from_utf8(buf[..error_len].to_vec())
                        .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in error: {}", e))?,
                )
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            msg_type,
            data,
            end,
            error,
        })
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
        self.cancel.cancel();
    }
}

/// Parser for binary stream frames
///
/// Accumulates bytes and yields complete frames as they become available.
#[derive(Default)]
pub struct BinaryFrameParser {
    buffer: BytesMut,
}

impl BinaryFrameParser {
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::new(),
        }
    }

    /// Add data to the buffer
    pub fn push(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Try to parse a complete frame from the buffer
    pub fn try_parse(&mut self) -> Option<anyhow::Result<StreamFrame>> {
        if self.buffer.len() < 4 {
            return None;
        }

        // Peek at length without consuming
        let total_len = u32::from_be_bytes([
            self.buffer[0],
            self.buffer[1],
            self.buffer[2],
            self.buffer[3],
        ]) as usize;

        // Check if we have the complete frame
        if self.buffer.len() < 4 + total_len {
            return None;
        }

        // Parse the frame
        let frame_bytes = self.buffer.split_to(4 + total_len);
        Some(StreamFrame::from_binary(&frame_bytes))
    }

    /// Parse all available frames
    pub fn parse_all(&mut self) -> Vec<anyhow::Result<StreamFrame>> {
        let mut frames = Vec::new();
        while let Some(result) = self.try_parse() {
            frames.push(result);
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_frame_data() {
        let frame = StreamFrame::data("test", b"hello");
        assert_eq!(frame.msg_type, "test");
        assert!(!frame.end);
        assert!(frame.error.is_none());
        assert_eq!(frame.get_data(), b"hello");
    }

    #[test]
    fn test_stream_frame_end() {
        let frame = StreamFrame::end();
        assert!(frame.end);
        assert!(frame.error.is_none());
    }

    #[test]
    fn test_stream_frame_error() {
        let frame = StreamFrame::error("something went wrong");
        assert!(frame.end);
        assert!(frame.is_error());
        assert_eq!(frame.error.as_ref().unwrap(), "something went wrong");
    }

    #[test]
    fn test_binary_roundtrip() {
        let frame = StreamFrame::data("token", b"hello world");
        let bytes = frame.to_binary();
        let parsed = StreamFrame::from_binary(&bytes).unwrap();

        assert_eq!(parsed.msg_type, "token");
        assert_eq!(parsed.get_data(), b"hello world");
        assert!(!parsed.end);
    }

    #[test]
    fn test_binary_end_frame() {
        let frame = StreamFrame::end();
        let bytes = frame.to_binary();
        let parsed = StreamFrame::from_binary(&bytes).unwrap();

        assert!(parsed.end);
        assert!(parsed.get_data().is_empty());
    }

    #[test]
    fn test_binary_error_frame() {
        let frame = StreamFrame::error("test error");
        let bytes = frame.to_binary();
        let parsed = StreamFrame::from_binary(&bytes).unwrap();

        assert!(parsed.end);
        assert!(parsed.is_error());
        assert_eq!(parsed.error.as_ref().unwrap(), "test error");
    }

    #[test]
    fn test_binary_large_payload() {
        let large_data = vec![0xABu8; 1024 * 1024]; // 1MB
        let frame = StreamFrame::data("large", &large_data);
        let bytes = frame.to_binary();
        let parsed = StreamFrame::from_binary(&bytes).unwrap();

        assert_eq!(parsed.get_data(), &large_data[..]);
    }

    #[test]
    fn test_binary_frame_parser() {
        let mut parser = BinaryFrameParser::new();

        let frame1 = StreamFrame::data("chunk", b"first");
        let frame2 = StreamFrame::data("chunk", b"second");
        let frame3 = StreamFrame::end();

        parser.push(&frame1.to_binary());
        parser.push(&frame2.to_binary());
        parser.push(&frame3.to_binary());

        let frames = parser.parse_all();
        assert_eq!(frames.len(), 3);

        assert_eq!(frames[0].as_ref().unwrap().get_data(), b"first");
        assert_eq!(frames[1].as_ref().unwrap().get_data(), b"second");
        assert!(frames[2].as_ref().unwrap().end);
    }

    #[test]
    fn test_binary_frame_parser_partial() {
        let mut parser = BinaryFrameParser::new();

        let frame = StreamFrame::data("test", b"payload");
        let bytes = frame.to_binary();

        // Push partial data
        parser.push(&bytes[..5]);
        assert!(parser.try_parse().is_none());

        // Push rest
        parser.push(&bytes[5..]);
        let parsed = parser.try_parse().unwrap().unwrap();
        assert_eq!(parsed.get_data(), b"payload");
    }

    #[test]
    fn test_from_message_with_msg_type() {
        let msg = Message::single("token", b"hello");
        let frame = StreamFrame::from_message(&msg, "default");

        assert_eq!(frame.msg_type, "token");
        assert_eq!(frame.get_data(), b"hello");
    }

    #[test]
    fn test_from_message_uses_default_msg_type() {
        let msg = Message::single("", b"hello");
        let frame = StreamFrame::from_message(&msg, "default_type");

        assert_eq!(frame.msg_type, "default_type");
    }

    #[test]
    fn test_to_message_roundtrip() {
        let original = Message::single("chunk", b"data");
        let frame = StreamFrame::from_message(&original, "default");
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
        let frame = StreamFrame::end();
        let msg = frame.to_message().unwrap();
        assert!(msg.is_none());
    }

    #[test]
    fn test_to_message_error_frame() {
        let frame = StreamFrame::error("test error");
        let result = frame.to_message();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test error"));
    }
}
