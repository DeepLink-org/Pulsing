//! Transport module - network communication layer
//!
//! Provides reliable message transport between nodes.
//!
//! ## Transport Implementations
//!
//! - `http`: HTTP/1.1 transport (legacy, for compatibility)
//! - `http2`: HTTP/2 transport with streaming support (recommended)
//! - `tcp`: Raw TCP transport
//!
//! ## HTTP/2 Features
//!
//! The HTTP/2 transport (`http2` module) provides:
//! - h2c (HTTP/2 over cleartext) - no TLS required
//! - Streaming responses via `ask_stream`
//! - Connection multiplexing
//! - Built-in flow control (backpressure)

pub mod codec;
pub mod http;
pub mod http2;
pub mod tcp;

// Legacy HTTP/1.1 exports
pub use codec::{MessageCodec, TransportMessage};
pub use http::{
    ActorRequest, ActorResponse, GossipRequest, GossipResponse, HttpMessageHandler,
    HttpRemoteTransport, HttpTransport, HttpTransportConfig,
};
pub use tcp::{TcpRemoteTransport, TcpTransport, TcpTransportConfig};

// HTTP/2 exports (default, recommended)
pub use http2::{
    Http2Client, Http2ClientBuilder, Http2Config, Http2RemoteTransport, Http2Server,
    Http2ServerHandler, Http2Transport, MessageMode, PoolConfig, PoolStats, RetryConfig,
    RetryableError, StreamFrame, StreamHandle,
};
