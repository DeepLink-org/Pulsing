//! HTTP/2 transport layer.

mod client;
mod config;
mod pool;
mod retry;
mod server;
mod stream;

use crate::error::{PulsingError, Result, RuntimeError};

#[cfg(feature = "tls")]
mod tls;

pub use client::{
    FaultInjectContext, FaultInjectOperation, FaultInjector, Http2Client, Http2ClientBuilder,
};
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
    ) -> crate::error::Result<(Arc<Self>, SocketAddr)> {
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
    pub async fn ask(&self, addr: SocketAddr, actor_name: &str, msg: Message) -> Result<Message> {
        let path = format!("/actors/{}", actor_name);
        self.client.send_message_full(addr, &path, msg).await
    }

    /// Send a request to a named actor and wait for response.
    pub async fn ask_named(
        &self,
        addr: SocketAddr,
        path: &ActorPath,
        msg: Message,
    ) -> Result<Message> {
        let url_path = format!("/named/{}", path.as_str());
        self.client.send_message_full(addr, &url_path, msg).await
    }

    pub async fn tell(&self, addr: SocketAddr, actor_name: &str, msg: Message) -> Result<()> {
        let path = format!("/actors/{}", actor_name);
        let Message::Single { msg_type, data } = msg else {
            return Err(RuntimeError::protocol_error("Streaming not supported for tell").into());
        };

        self.client.tell(addr, &path, &msg_type, data).await
    }

    pub async fn tell_named(&self, addr: SocketAddr, path: &ActorPath, msg: Message) -> Result<()> {
        let url_path = format!("/named/{}", path.as_str());
        let Message::Single { msg_type, data } = msg else {
            return Err(RuntimeError::protocol_error("Streaming not supported for tell").into());
        };

        self.client.tell(addr, &url_path, &msg_type, data).await
    }

    /// Send a gossip message
    pub async fn send_gossip(&self, addr: SocketAddr, payload: Vec<u8>) -> Result<Option<Vec<u8>>> {
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

/// Describes which actor on the remote node to reach.
#[derive(Debug, Clone)]
pub enum TransportTarget {
    /// Actor registered by name (path: `/actors/{name}`)
    ByName(String),
    /// Actor identified by its UUID (path: `/actors/{id}`)
    ById(ActorId),
    /// Actor registered under a hierarchical named path (path: `/named/{path}`)
    Named(ActorPath),
}

impl TransportTarget {
    fn to_path(&self) -> String {
        match self {
            TransportTarget::ByName(name) => format!("/actors/{}", name),
            TransportTarget::ById(id) => format!("/actors/{}", id),
            TransportTarget::Named(path) => format!("/named/{}", path.as_str()),
        }
    }
}

/// Builder for [`Http2RemoteTransport`].
///
/// ```ignore
/// let transport = Http2RemoteTransport::builder(
///     client, addr, TransportTarget::ByName("my_actor".into()))
///     .circuit_breaker(CircuitBreakerConfig::default())
///     .build();
/// ```
pub struct Http2RemoteTransportBuilder {
    client: Arc<Http2Client>,
    remote_addr: SocketAddr,
    target: TransportTarget,
    cb_config: Option<CircuitBreakerConfig>,
}

impl Http2RemoteTransportBuilder {
    pub fn circuit_breaker(mut self, config: CircuitBreakerConfig) -> Self {
        self.cb_config = Some(config);
        self
    }

    pub fn build(self) -> Http2RemoteTransport {
        Http2RemoteTransport {
            client: self.client,
            remote_addr: self.remote_addr,
            path: self.target.to_path(),
            circuit_breaker: match self.cb_config {
                Some(cfg) => CircuitBreaker::with_config(cfg),
                None => CircuitBreaker::new(),
            },
        }
    }
}

/// HTTP/2 Remote Transport for ActorRef
///
/// Implements the `RemoteTransport` trait, enabling `ActorRef` to communicate
/// with remote actors over HTTP/2, including streaming support.
///
/// Build via [`Http2RemoteTransport::builder`]:
/// ```ignore
/// let transport = Http2RemoteTransport::builder(
///     client, addr, TransportTarget::ByName("my_actor".into())).build();
/// ```
pub struct Http2RemoteTransport {
    client: Arc<Http2Client>,
    remote_addr: SocketAddr,
    /// Request path (e.g., "/actors/name" or "/named/path")
    path: String,
    circuit_breaker: CircuitBreaker,
}

impl Http2RemoteTransport {
    /// Start building a transport for the given target.
    pub fn builder(
        client: Arc<Http2Client>,
        remote_addr: SocketAddr,
        target: TransportTarget,
    ) -> Http2RemoteTransportBuilder {
        Http2RemoteTransportBuilder {
            client,
            remote_addr,
            target,
            cb_config: None,
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
    ) -> Result<Vec<u8>> {
        if !self.circuit_breaker.can_execute() {
            return Err(PulsingError::from(RuntimeError::ConnectionFailed {
                addr: self.remote_addr.to_string(),
                reason: "Circuit breaker is open".to_string(),
            }));
        }

        let result = self
            .client
            .ask(self.remote_addr, &self.path, msg_type, payload)
            .await
            .map_err(|e| PulsingError::from(RuntimeError::Other(e.to_string())));

        self.circuit_breaker.record_outcome(result.is_ok());
        result
    }

    async fn send(&self, _actor_id: &ActorId, msg_type: &str, payload: Vec<u8>) -> Result<()> {
        if !self.circuit_breaker.can_execute() {
            return Err(PulsingError::from(RuntimeError::ConnectionFailed {
                addr: self.remote_addr.to_string(),
                reason: "Circuit breaker is open".to_string(),
            }));
        }

        let result = self
            .client
            .tell(self.remote_addr, &self.path, msg_type, payload)
            .await
            .map_err(|e| PulsingError::from(RuntimeError::Other(e.to_string())));

        self.circuit_breaker.record_outcome(result.is_ok());
        result
    }

    /// Send a message and receive response (unified interface)
    async fn send_message(&self, _actor_id: &ActorId, msg: Message) -> Result<Message> {
        if !self.circuit_breaker.can_execute() {
            return Err(PulsingError::from(RuntimeError::ConnectionFailed {
                addr: self.remote_addr.to_string(),
                reason: "Circuit breaker is open".to_string(),
            }));
        }

        let result = self
            .client
            .send_message_full(self.remote_addr, &self.path, msg)
            .await
            .map_err(|e| PulsingError::from(RuntimeError::Other(e.to_string())));

        self.circuit_breaker.record_outcome(result.is_ok());
        result
    }

    /// Send a one-way message (unified interface)
    async fn send_oneway(&self, actor_id: &ActorId, msg: Message) -> Result<()> {
        let Message::Single { msg_type, data } = msg else {
            return Err(PulsingError::from(RuntimeError::Other(
                "Streaming not supported for fire-and-forget (use ask pattern instead)".into(),
            )));
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

        let transport = Http2RemoteTransport::builder(
            client.clone(),
            addr,
            TransportTarget::ByName("my_actor".into()),
        )
        .build();
        assert_eq!(transport.path(), "/actors/my_actor");

        let path = ActorPath::new("services/llm").unwrap();
        let transport =
            Http2RemoteTransport::builder(client, addr, TransportTarget::Named(path)).build();
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

        let transport =
            Http2RemoteTransport::builder(client, addr, TransportTarget::ById(actor_id)).build();
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

        let transport =
            Http2RemoteTransport::builder(client, addr, TransportTarget::ByName("my_actor".into()))
                .circuit_breaker(cb_config)
                .build();
        assert_eq!(transport.path(), "/actors/my_actor");
        assert!(transport.circuit_breaker().can_execute());
    }

    #[test]
    fn test_http2_remote_transport_named_with_circuit_breaker() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let path = ActorPath::new("services/llm").unwrap();
        let cb_config = CircuitBreakerConfig::default();

        let transport = Http2RemoteTransport::builder(client, addr, TransportTarget::Named(path))
            .circuit_breaker(cb_config)
            .build();
        assert_eq!(transport.path(), "/named/services/llm");
    }

    #[test]
    fn test_http2_remote_transport_accessors() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "192.168.1.100:9000".parse().unwrap();

        let transport = Http2RemoteTransport::builder(
            client.clone(),
            addr,
            TransportTarget::ByName("test_actor".into()),
        )
        .build();

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
