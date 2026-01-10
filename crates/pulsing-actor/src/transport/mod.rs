//! Transport module - network communication layer
//!
//! Provides reliable message transport between nodes using HTTP/2.
//!
//! ## HTTP/2 Transport
//!
//! The HTTP/2 transport provides:
//! - h2c (HTTP/2 over cleartext) - no TLS required
//! - Streaming responses for efficient data transfer
//! - Connection multiplexing with advanced pooling
//! - Built-in flow control (backpressure)
//! - Retry strategies with exponential backoff
//! - Circuit breaker for fault tolerance

pub mod http2;

// HTTP/2 exports
pub use http2::{
    BinaryFrameParser, Http2Client, Http2ClientBuilder, Http2Config, Http2RemoteTransport,
    Http2Server, Http2ServerHandler, Http2Transport, MessageMode, PoolConfig, PoolStats,
    RequestType, RetryConfig, RetryableError, StreamFrame, StreamHandle, FLAG_END, FLAG_ERROR,
};
