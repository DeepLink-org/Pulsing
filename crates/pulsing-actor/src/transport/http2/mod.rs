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

use crate::actor::{ActorId, ActorPath, Message, RemoteTransport};
use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// High-level HTTP/2 Transport
///
/// Combines Http2Server and Http2Client into a single component
/// used by ActorSystem and GossipCluster.
pub struct Http2Transport {
    local_addr: SocketAddr,
    client: Arc<Http2Client>,
    // server: Option<Http2Server>,
    // config: Http2Config,
}

impl Http2Transport {
    /// Create a new HTTP/2 transport and start the server
    pub async fn new(
        bind_addr: SocketAddr,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) -> anyhow::Result<(Arc<Self>, SocketAddr)> {
        // Build HTTP/2 client
        let client = Arc::new(Http2Client::new(config.clone()));
        client.start_background_tasks();

        // Start HTTP/2 server
        let server = Http2Server::new(bind_addr, handler, config.clone(), cancel.clone()).await?;
        let local_addr = server.local_addr();

        let transport = Arc::new(Self {
            local_addr,
            client,
            // server: Some(server),
            // config,
        });

        Ok((transport, local_addr))
    }

    /// Create a client-only transport (no server)
    pub fn new_client(config: Http2Config) -> Arc<Self> {
        let client = Arc::new(Http2Client::new(config.clone()));
        client.start_background_tasks();

        Arc::new(Self {
            local_addr: "0.0.0.0:0".parse().unwrap(),
            client,
            // server: None,
            // config,
        })
    }

    /// Get local address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Send a request to an actor and wait for response
    pub async fn ask(
        &self,
        addr: SocketAddr,
        actor_name: &str,
        msg: Message,
    ) -> anyhow::Result<Message> {
        let path = format!("/actors/{}", actor_name);
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming requests not yet supported"));
        };

        let response = self.client.ask(addr, &path, &msg_type, data).await?;
        Ok(Message::single("", response))
    }

    /// Send a request to a named actor and wait for response
    pub async fn ask_named(
        &self,
        addr: SocketAddr,
        path: &ActorPath,
        msg: Message,
    ) -> anyhow::Result<Message> {
        let url_path = format!("/named/{}", path.as_str());
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming requests not yet supported"));
        };

        let response = self.client.ask(addr, &url_path, &msg_type, data).await?;
        Ok(Message::single("", response))
    }

    /// Send a fire-and-forget message
    pub async fn tell(
        &self,
        addr: SocketAddr,
        actor_name: &str,
        msg: Message,
    ) -> anyhow::Result<()> {
        let path = format!("/actors/{}", actor_name);
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming not supported for tell"));
        };

        self.client.tell(addr, &path, &msg_type, data).await
    }

    /// Send a fire-and-forget message to a named actor
    pub async fn tell_named(
        &self,
        addr: SocketAddr,
        path: &ActorPath,
        msg: Message,
    ) -> anyhow::Result<()> {
        let url_path = format!("/named/{}", path.as_str());
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming not supported for tell"));
        };

        self.client.tell(addr, &url_path, &msg_type, data).await
    }

    /// Send a gossip message
    pub async fn send_gossip(
        &self,
        addr: SocketAddr,
        payload: Vec<u8>,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let response = self
            .client
            .ask(addr, "/cluster/gossip", "gossip", payload)
            .await?;

        if response.is_empty() {
            Ok(None)
        } else {
            Ok(Some(response))
        }
    }

    /// Get the underlying client
    pub fn client(&self) -> Arc<Http2Client> {
        self.client.clone()
    }
}

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

    pub fn parse(s: &str) -> Option<Self> {
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
    pub const RESPONSE_TYPE: &str = "x-response-type";
    pub const REQUEST_ID: &str = "x-request-id";
}

/// Response type for unified message handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseType {
    Single,
    Stream,
}

impl ResponseType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResponseType::Single => "single",
            ResponseType::Stream => "stream",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "single" => Some(ResponseType::Single),
            "stream" => Some(ResponseType::Stream),
            _ => None,
        }
    }
}

/// HTTP/2 Remote Transport for ActorRef
///
/// Implements the `RemoteTransport` trait, enabling `ActorRef` to communicate
/// with remote actors over HTTP/2, including streaming support.
///
/// Features:
/// - Automatic connection pooling and reuse
/// - Retry with exponential backoff for transient failures
/// - Circuit breaker for fault tolerance
/// - Configurable timeouts
/// - Streaming response support
pub struct Http2RemoteTransport {
    client: Arc<Http2Client>,
    remote_addr: SocketAddr,
    /// Request path (e.g., "/actors/name" or "/named/path")
    path: String,
    circuit_breaker: CircuitBreaker,
}

impl Http2RemoteTransport {
    /// Create a new remote transport targeting an actor by name
    pub fn new(client: Arc<Http2Client>, remote_addr: SocketAddr, actor_name: String) -> Self {
        Self {
            client,
            remote_addr,
            path: format!("/actors/{}", actor_name),
            circuit_breaker: CircuitBreaker::new(),
        }
    }

    /// Create a new remote transport targeting an actor by ID
    pub fn new_by_id(client: Arc<Http2Client>, remote_addr: SocketAddr, actor_id: ActorId) -> Self {
        Self {
            client,
            remote_addr,
            path: format!("/actors/{}", actor_id.local_id()),
            circuit_breaker: CircuitBreaker::new(),
        }
    }

    /// Create a new remote transport targeting a named actor by path
    pub fn new_named(client: Arc<Http2Client>, remote_addr: SocketAddr, path: ActorPath) -> Self {
        Self {
            client,
            remote_addr,
            path: format!("/named/{}", path.as_str()),
            circuit_breaker: CircuitBreaker::new(),
        }
    }

    /// Create a new remote transport with custom circuit breaker configuration
    pub fn with_circuit_breaker(
        client: Arc<Http2Client>,
        remote_addr: SocketAddr,
        actor_name: String,
        cb_config: CircuitBreakerConfig,
    ) -> Self {
        Self {
            client,
            remote_addr,
            path: format!("/actors/{}", actor_name),
            circuit_breaker: CircuitBreaker::with_config(cb_config),
        }
    }

    /// Create a new remote transport targeting a named actor with custom circuit breaker
    pub fn new_named_with_circuit_breaker(
        client: Arc<Http2Client>,
        remote_addr: SocketAddr,
        path: ActorPath,
        cb_config: CircuitBreakerConfig,
    ) -> Self {
        Self {
            client,
            remote_addr,
            path: format!("/named/{}", path.as_str()),
            circuit_breaker: CircuitBreaker::with_config(cb_config),
        }
    }

    /// Get the circuit breaker (for monitoring/debugging)
    pub fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }

    /// Get the underlying HTTP/2 client
    pub fn client(&self) -> &Arc<Http2Client> {
        &self.client
    }

    /// Get the remote address
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// Get the request path
    pub fn path(&self) -> &str {
        &self.path
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
        // Check circuit breaker before making request
        if !self.circuit_breaker.can_execute() {
            return Err(anyhow::anyhow!(
                "Circuit breaker is open for {}",
                self.remote_addr
            ));
        }

        let result = self
            .client
            .ask(self.remote_addr, &self.path, msg_type, payload)
            .await;

        // Record outcome in circuit breaker
        self.circuit_breaker.record_outcome(result.is_ok());
        result
    }

    async fn send(
        &self,
        _actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        // Check circuit breaker before making request
        if !self.circuit_breaker.can_execute() {
            return Err(anyhow::anyhow!(
                "Circuit breaker is open for {}",
                self.remote_addr
            ));
        }

        let result = self
            .client
            .tell(self.remote_addr, &self.path, msg_type, payload)
            .await;

        // Record outcome in circuit breaker
        self.circuit_breaker.record_outcome(result.is_ok());
        result
    }

    /// Send a message and receive response (unified interface)
    ///
    /// This method is the primary way ActorRef communicates with remote actors.
    /// It automatically handles both single and stream responses based on
    /// the server's response type header.
    async fn send_message(&self, _actor_id: &ActorId, msg: Message) -> anyhow::Result<Message> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming requests not yet supported"));
        };
        // Use unified send_message that auto-detects response type
        self.client
            .send_message(self.remote_addr, &self.path, &msg_type, data)
            .await
    }

    /// Send a one-way message (unified interface)
    async fn send_oneway(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<()> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!(
                "Streaming not supported for fire-and-forget"
            ));
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

        assert_eq!(MessageMode::parse("ask"), Some(MessageMode::Ask));
        assert_eq!(MessageMode::parse("TELL"), Some(MessageMode::Tell));
        assert_eq!(MessageMode::parse("Stream"), Some(MessageMode::Stream));
        assert_eq!(MessageMode::parse("invalid"), None);
    }

    #[test]
    fn test_request_path() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let transport = Http2RemoteTransport::new(client.clone(), addr, "my_actor".to_string());
        assert_eq!(transport.path(), "/actors/my_actor");

        let path = ActorPath::new("services/llm").unwrap();
        let transport = Http2RemoteTransport::new_named(client, addr, path);
        assert_eq!(transport.path(), "/named/services/llm");
    }
}
