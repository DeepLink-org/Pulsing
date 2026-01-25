//! HTTP/2 transport layer.

mod client;
mod config;
mod pool;
mod retry;
mod server;
mod stream;

#[cfg(feature = "tls")]
mod tls;

pub use client::{Http2Client, Http2ClientBuilder};
pub use config::Http2Config;
pub use pool::{ConnectionPool, PoolConfig, PoolStats};
pub use retry::{RetryConfig, RetryExecutor, RetryableError};
pub use server::{Http2Server, Http2ServerHandler};
pub use stream::{BinaryFrameParser, StreamFrame, StreamHandle, FLAG_END, FLAG_ERROR};

#[cfg(feature = "tls")]
pub use tls::TlsConfig;

use crate::actor::{ActorId, ActorPath, Message, RemoteTransport};
use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// High-level HTTP/2 transport.
pub struct Http2Transport {
    local_addr: SocketAddr,
    client: Arc<Http2Client>,
    // server: Option<Http2Server>,
    // config: Http2Config,
}

impl Http2Transport {
    pub async fn new(
        bind_addr: SocketAddr,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) -> anyhow::Result<(Arc<Self>, SocketAddr)> {
        let client = Arc::new(Http2Client::new(config.clone()));
        client.start_background_tasks();

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

    pub fn new_client(config: Http2Config) -> Arc<Self> {
        let client = Arc::new(Http2Client::new(config.clone()));
        client.start_background_tasks();

        const CLIENT_ONLY_ADDR: SocketAddr =
            SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)), 0);

        Arc::new(Self {
            local_addr: CLIENT_ONLY_ADDR,
            client,
            // server: None,
            // config,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Send a request to an actor and wait for response.
    pub async fn ask(
        &self,
        addr: SocketAddr,
        actor_name: &str,
        msg: Message,
    ) -> anyhow::Result<Message> {
        let path = format!("/actors/{}", actor_name);
        self.client.send_message_full(addr, &path, msg).await
    }

    /// Send a request to a named actor and wait for response.
    pub async fn ask_named(
        &self,
        addr: SocketAddr,
        path: &ActorPath,
        msg: Message,
    ) -> anyhow::Result<Message> {
        let url_path = format!("/named/{}", path.as_str());
        self.client.send_message_full(addr, &url_path, msg).await
    }

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
    pub const REQUEST_TYPE: &str = "x-request-type";
    pub const REQUEST_ID: &str = "x-request-id";
}

/// Request type for unified message handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    /// Single data payload
    Single,
    /// Streaming request (binary frames)
    Stream,
}

impl RequestType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestType::Single => "single",
            RequestType::Stream => "stream",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "single" => Some(RequestType::Single),
            "stream" => Some(RequestType::Stream),
            _ => None,
        }
    }
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
            path: format!("/actors/{}", actor_id),
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
    /// It automatically handles both:
    /// - Single and streaming requests (based on Message type)
    /// - Single and streaming responses (based on server's response type header)
    async fn send_message(&self, _actor_id: &ActorId, msg: Message) -> anyhow::Result<Message> {
        // Check circuit breaker before making request
        if !self.circuit_breaker.can_execute() {
            return Err(anyhow::anyhow!(
                "Circuit breaker is open for {}",
                self.remote_addr
            ));
        }

        // Use unified send_message_full that handles both single and streaming
        let result = self
            .client
            .send_message_full(self.remote_addr, &self.path, msg)
            .await;

        // Record outcome in circuit breaker
        self.circuit_breaker.record_outcome(result.is_ok());
        result
    }

    /// Send a one-way message (unified interface)
    ///
    /// Note: Streaming requests are NOT supported for fire-and-forget messages
    /// because there's no way to know when the stream is fully consumed.
    async fn send_oneway(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<()> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!(
                "Streaming not supported for fire-and-forget (use ask pattern instead)"
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

    #[test]
    fn test_message_mode_equality() {
        assert_eq!(MessageMode::Ask, MessageMode::Ask);
        assert_eq!(MessageMode::Tell, MessageMode::Tell);
        assert_eq!(MessageMode::Stream, MessageMode::Stream);
        assert_ne!(MessageMode::Ask, MessageMode::Tell);
        assert_ne!(MessageMode::Ask, MessageMode::Stream);
        assert_ne!(MessageMode::Tell, MessageMode::Stream);
    }

    #[test]
    fn test_message_mode_parse_case_insensitive() {
        assert_eq!(MessageMode::parse("ASK"), Some(MessageMode::Ask));
        assert_eq!(MessageMode::parse("Ask"), Some(MessageMode::Ask));
        assert_eq!(MessageMode::parse("aSk"), Some(MessageMode::Ask));
        assert_eq!(MessageMode::parse("STREAM"), Some(MessageMode::Stream));
    }

    #[test]
    fn test_message_mode_parse_empty() {
        assert_eq!(MessageMode::parse(""), None);
    }

    #[test]
    fn test_response_type() {
        assert_eq!(ResponseType::Single.as_str(), "single");
        assert_eq!(ResponseType::Stream.as_str(), "stream");

        assert_eq!(ResponseType::parse("single"), Some(ResponseType::Single));
        assert_eq!(ResponseType::parse("stream"), Some(ResponseType::Stream));
        assert_eq!(ResponseType::parse("SINGLE"), Some(ResponseType::Single));
        assert_eq!(ResponseType::parse("STREAM"), Some(ResponseType::Stream));
        assert_eq!(ResponseType::parse("invalid"), None);
        assert_eq!(ResponseType::parse(""), None);
    }

    #[test]
    fn test_response_type_equality() {
        assert_eq!(ResponseType::Single, ResponseType::Single);
        assert_eq!(ResponseType::Stream, ResponseType::Stream);
        assert_ne!(ResponseType::Single, ResponseType::Stream);
    }

    #[test]
    fn test_http2_headers() {
        assert_eq!(headers::MESSAGE_MODE, "x-message-mode");
        assert_eq!(headers::MESSAGE_TYPE, "x-message-type");
        assert_eq!(headers::RESPONSE_TYPE, "x-response-type");
        assert_eq!(headers::REQUEST_ID, "x-request-id");
    }

    #[test]
    fn test_http2_remote_transport_new_by_id() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let actor_id = ActorId::generate();

        let transport = Http2RemoteTransport::new_by_id(client, addr, actor_id);
        // Path should be /actors/{uuid} where uuid is 32 hex chars
        assert!(transport.path().starts_with("/actors/"));
        assert_eq!(transport.path().len(), 8 + 32); // "/actors/" + 32 hex chars
        assert_eq!(transport.remote_addr(), addr);
    }

    #[test]
    fn test_http2_remote_transport_with_circuit_breaker() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let cb_config = CircuitBreakerConfig::default();

        let transport = Http2RemoteTransport::with_circuit_breaker(
            client,
            addr,
            "my_actor".to_string(),
            cb_config,
        );
        assert_eq!(transport.path(), "/actors/my_actor");
        assert!(transport.circuit_breaker().can_execute());
    }

    #[test]
    fn test_http2_remote_transport_named_with_circuit_breaker() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let path = ActorPath::new("services/llm").unwrap();
        let cb_config = CircuitBreakerConfig::default();

        let transport =
            Http2RemoteTransport::new_named_with_circuit_breaker(client, addr, path, cb_config);
        assert_eq!(transport.path(), "/named/services/llm");
    }

    #[test]
    fn test_http2_remote_transport_accessors() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "192.168.1.100:9000".parse().unwrap();

        let transport = Http2RemoteTransport::new(client.clone(), addr, "test_actor".to_string());

        assert_eq!(transport.remote_addr(), addr);
        assert_eq!(transport.path(), "/actors/test_actor");
        assert!(Arc::ptr_eq(transport.client(), &client));
    }

    #[tokio::test]
    async fn test_http2_transport_new_client() {
        let config = Http2Config::default();
        let transport = Http2Transport::new_client(config);

        assert_eq!(transport.local_addr().port(), 0);
    }
}
