//! HTTP/2 Transport Layer
//!
//! Provides HTTP/2 (h2c - cleartext) transport for actor communication with streaming support.
//!
//! ## Features
//!
//! - HTTP/2 over cleartext (h2c) - no TLS required
//! - Streaming responses via `ask_stream`
//! - Connection multiplexing with advanced pooling
//! - Retry strategies with exponential backoff
//! - Timeout management at multiple levels
//! - Built-in flow control (backpressure)
//!
//! ## Protocol
//!
//! ### Message Modes
//!
//! - `ask`: Request-response pattern
//! - `tell`: Fire-and-forget pattern
//! - `stream`: Streaming response pattern
//!
//! ### Headers
//!
//! - `x-message-mode`: ask | tell | stream
//! - `x-message-type`: Message type identifier
//! - `x-request-id`: Optional request ID for tracing
//!
//! ## Example
//!
//! ```rust,ignore
//! use pulsing_actor::transport::http2::{Http2Client, Http2ClientBuilder, Http2Config};
//! use std::time::Duration;
//!
//! // Create client with custom configuration
//! let client = Http2ClientBuilder::new()
//!     .max_retries(3)
//!     .connect_timeout(Duration::from_secs(5))
//!     .request_timeout(Duration::from_secs(30))
//!     .build();
//!
//! // Send request
//! let response = client.ask(addr, "/actors/my_actor", "Ping", payload).await?;
//!
//! // Streaming request
//! let stream = client.ask_stream(addr, "/actors/my_actor", "StreamingRequest", payload).await?;
//! while let Some(frame) = stream.next().await {
//!     // Process streaming frames
//! }
//! ```

mod client;
mod config;
mod pool;
mod retry;
mod server;
mod stream;

pub use client::{Http2Client, Http2ClientBuilder};
pub use config::Http2Config;
pub use pool::{ConnectionPool, PoolConfig, PoolStats};
pub use retry::{RetryConfig, RetryExecutor, RetryableError};
pub use server::{Http2Server, Http2ServerHandler};
pub use stream::{StreamFrame, StreamHandle};

use crate::actor::{ActorId, ActorPath, Message, PayloadStream, RemoteTransport};
use futures::StreamExt;
use std::net::SocketAddr;
use std::sync::Arc;

/// Message mode for HTTP/2 requests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageMode {
    /// Request-response pattern
    Ask,
    /// Fire-and-forget pattern
    Tell,
    /// Streaming response pattern
    Stream,
}

impl MessageMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageMode::Ask => "ask",
            MessageMode::Tell => "tell",
            MessageMode::Stream => "stream",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ask" => Some(MessageMode::Ask),
            "tell" => Some(MessageMode::Tell),
            "stream" => Some(MessageMode::Stream),
            _ => None,
        }
    }
}

/// HTTP header names
pub mod headers {
    pub const MESSAGE_MODE: &str = "x-message-mode";
    pub const MESSAGE_TYPE: &str = "x-message-type";
    pub const REQUEST_ID: &str = "x-request-id";
}

/// Target type for HTTP/2 remote transport
#[derive(Clone)]
enum Http2RemoteTarget {
    /// Target by actor name
    Actor(String),
    /// Target by named actor path
    Named(ActorPath),
}

/// HTTP/2 Remote Transport for ActorRef
///
/// Implements the `RemoteTransport` trait, enabling `ActorRef` to communicate
/// with remote actors over HTTP/2, including streaming support.
///
/// Features:
/// - Automatic connection pooling and reuse
/// - Retry with exponential backoff for transient failures
/// - Configurable timeouts
/// - Streaming response support
pub struct Http2RemoteTransport {
    client: Arc<Http2Client>,
    remote_addr: SocketAddr,
    target: Http2RemoteTarget,
}

impl Http2RemoteTransport {
    /// Create a new remote transport targeting an actor by name
    pub fn new(client: Arc<Http2Client>, remote_addr: SocketAddr, actor_name: String) -> Self {
        Self {
            client,
            remote_addr,
            target: Http2RemoteTarget::Actor(actor_name),
        }
    }

    /// Create a new remote transport targeting a named actor by path
    pub fn new_named(client: Arc<Http2Client>, remote_addr: SocketAddr, path: ActorPath) -> Self {
        Self {
            client,
            remote_addr,
            target: Http2RemoteTarget::Named(path),
        }
    }

    /// Get the underlying HTTP/2 client
    pub fn client(&self) -> &Arc<Http2Client> {
        &self.client
    }

    /// Get the remote address
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// Get the path for the request
    fn request_path(&self) -> String {
        match &self.target {
            Http2RemoteTarget::Actor(name) => format!("/actors/{}", name),
            Http2RemoteTarget::Named(path) => format!("/named/{}", path.as_str()),
        }
    }
}

#[async_trait::async_trait]
impl RemoteTransport for Http2RemoteTransport {
    async fn request(
        &self,
        _actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        let path = self.request_path();
        self.client.ask(self.remote_addr, &path, msg_type, payload).await
    }

    async fn send(
        &self,
        _actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        let path = self.request_path();
        self.client.tell(self.remote_addr, &path, msg_type, payload).await
    }

    async fn request_stream(
        &self,
        _actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<PayloadStream> {
        let path = self.request_path();
        let msg_stream = self
            .client
            .ask_stream_raw(self.remote_addr, &path, msg_type, payload)
            .await?;
        
        // Convert MessageStream to PayloadStream by extracting payload
        let payload_stream = msg_stream.map(|result| {
            result.and_then(|msg| {
                let Message::Single { data, .. } = msg else {
                    return Err(anyhow::anyhow!("Expected single message in stream"));
                };
                Ok(data)
            })
        });
        Ok(Box::pin(payload_stream))
    }

    /// Send a message and receive response (unified interface)
    ///
    /// This method is the primary way ActorRef communicates with remote actors.
    /// It automatically handles:
    /// - Connection pooling
    /// - Retry logic
    /// - Timeout management
    async fn send_message(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<Message> {
        let Message::Single { msg_type, data } = msg else {
            // For streaming requests, we need to use a different approach
            return Err(anyhow::anyhow!("Streaming requests require request_stream method"));
        };
        let response = self.request(actor_id, &msg_type, data).await?;
        Ok(Message::single("", response))
    }

    /// Send a one-way message (unified interface)
    async fn send_oneway(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<()> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming not supported for fire-and-forget"));
        };
        self.send(actor_id, &msg_type, data).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_mode() {
        assert_eq!(MessageMode::Ask.as_str(), "ask");
        assert_eq!(MessageMode::Tell.as_str(), "tell");
        assert_eq!(MessageMode::Stream.as_str(), "stream");

        assert_eq!(MessageMode::from_str("ask"), Some(MessageMode::Ask));
        assert_eq!(MessageMode::from_str("TELL"), Some(MessageMode::Tell));
        assert_eq!(MessageMode::from_str("Stream"), Some(MessageMode::Stream));
        assert_eq!(MessageMode::from_str("invalid"), None);
    }

    #[test]
    fn test_request_path() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let transport = Http2RemoteTransport::new(client.clone(), addr, "my_actor".to_string());
        assert_eq!(transport.request_path(), "/actors/my_actor");

        let path = ActorPath::new("services/llm").unwrap();
        let transport = Http2RemoteTransport::new_named(client, addr, path);
        assert_eq!(transport.request_path(), "/named/services/llm");
    }
}
