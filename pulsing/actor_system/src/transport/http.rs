//! HTTP-based transport layer
//!
//! Provides a unified HTTP transport for both actor messages and gossip protocol.
//! This simplifies network configuration by using a single port for all communication.
//!
//! ## Routes
//!
//! - `POST /actors/{name}` - Ask: send message and wait for response
//! - `PUT /actors/{name}` - Tell: fire-and-forget message
//! - `GET /actors/{name}` - Get actor info
//! - `POST /named/{path...}` - Ask: send message to named actor
//! - `PUT /named/{path...}` - Tell: fire-and-forget to named actor
//! - `GET /named/{path...}` - Get named actor info
//! - `POST /cluster/gossip` - Gossip protocol
//! - `GET /health` - Node health check

use crate::actor::{ActorId, ActorPath, Message, PayloadStream};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// HTTP transport configuration
#[derive(Clone, Debug)]
pub struct HttpTransportConfig {
    /// Request timeout
    pub request_timeout: Duration,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Keep-alive timeout
    pub keepalive_timeout: Duration,

    /// Max connections per host
    pub max_connections_per_host: usize,
}

impl Default for HttpTransportConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
            keepalive_timeout: Duration::from_secs(60),
            max_connections_per_host: 32,
        }
    }
}

/// Actor message request (wire format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorRequest {
    /// Message type
    pub msg_type: String,

    /// Serialized payload (base64 encoded for JSON transport)
    pub payload: Vec<u8>,

    /// Whether this is a one-way message (no response expected)
    pub oneway: bool,

    /// Request ID for correlation
    pub request_id: u64,
}

/// Actor message response (wire format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorResponse {
    /// Request ID for correlation
    pub request_id: u64,

    /// Response payload or error
    pub result: Result<Vec<u8>, String>,
}

/// Gossip protocol request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipRequest {
    /// Serialized gossip message
    pub payload: Vec<u8>,
}

/// Gossip protocol response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipResponse {
    /// Optional response payload
    pub payload: Option<Vec<u8>>,
}

/// Handler trait for processing incoming messages
#[async_trait::async_trait]
pub trait HttpMessageHandler: Send + Sync + 'static {
    /// Handle an actor message with ask pattern (expects response)
    async fn handle_actor_message(&self, actor_name: &str, msg: Message)
        -> anyhow::Result<Message>;

    /// Handle an actor message with tell pattern (fire-and-forget)
    async fn handle_actor_tell(&self, actor_name: &str, msg: Message) -> anyhow::Result<()>;

    /// Handle a named actor message with ask pattern (expects response)
    async fn handle_named_actor_message(&self, path: &str, msg: Message)
        -> anyhow::Result<Message>;

    /// Handle a named actor message with tell pattern (fire-and-forget)
    async fn handle_named_actor_tell(&self, path: &str, msg: Message) -> anyhow::Result<()>;

    /// Handle a gossip message
    async fn handle_gossip_message(&self, payload: Vec<u8>) -> anyhow::Result<Option<Vec<u8>>>;

    /// Get node info for health check
    async fn get_node_info(&self) -> serde_json::Value;

    /// Get specific actor info
    async fn get_actor_info(&self, actor_name: &str) -> Option<serde_json::Value>;

    /// Get named actor info
    async fn get_named_actor_info(&self, path: &str) -> Option<serde_json::Value>;
}

/// Shared state for HTTP handlers
struct HttpState {
    handler: Arc<dyn HttpMessageHandler>,
    #[allow(dead_code)]
    pending_requests: Arc<DashMap<u64, oneshot::Sender<Result<Vec<u8>, String>>>>,
}

/// HTTP Transport - unified transport for actor and gossip communication
pub struct HttpTransport {
    local_addr: SocketAddr,
    #[allow(dead_code)]
    config: HttpTransportConfig,
    client: reqwest::Client,
    request_id: AtomicU64,
    #[allow(dead_code)]
    cancel: CancellationToken,
}

impl HttpTransport {
    /// Create a new HTTP transport and start the server
    pub async fn new(
        bind_addr: SocketAddr,
        handler: Arc<dyn HttpMessageHandler>,
        config: HttpTransportConfig,
        cancel: CancellationToken,
    ) -> anyhow::Result<(Arc<Self>, SocketAddr)> {
        // Build HTTP client with connection pooling
        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .pool_idle_timeout(config.keepalive_timeout)
            .pool_max_idle_per_host(config.max_connections_per_host)
            .build()?;

        // Bind listener to get actual address
        let listener = TcpListener::bind(bind_addr).await?;
        let local_addr = listener.local_addr()?;

        let transport = Arc::new(Self {
            local_addr,
            config,
            client,
            request_id: AtomicU64::new(0),
            cancel: cancel.clone(),
        });

        // Build router
        let state = Arc::new(HttpState {
            handler,
            pending_requests: Arc::new(DashMap::new()),
        });

        let app = Router::new()
            .route("/health", get(health_handler))
            .route(
                "/actors/{name}",
                post(actor_handler)
                    .put(actor_tell_handler)
                    .get(actor_info_handler),
            )
            // Named actor routes - support variable path depth
            .route(
                "/named/{*path}",
                post(named_actor_handler)
                    .put(named_actor_tell_handler)
                    .get(named_actor_info_handler),
            )
            .route("/cluster/gossip", post(gossip_handler))
            .route("/cluster/registry", get(registry_handler))
            .with_state(state);

        // Start server
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    cancel_clone.cancelled().await;
                })
                .await
                .ok();
        });

        tracing::info!(addr = %local_addr, "HTTP transport started");

        Ok((transport, local_addr))
    }

    /// Get local address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Generate next request ID
    fn next_request_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a request to an actor and wait for response
    pub async fn send_request(
        &self,
        addr: SocketAddr,
        actor_name: &str,
        msg: Message,
    ) -> anyhow::Result<Message> {
        let request_id = self.next_request_id();
        let url = format!("http://{}/actors/{}", addr, actor_name);

        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!(
                "Streaming not supported by HTTP/1.1 transport"
            ));
        };
        let request = ActorRequest {
            msg_type,
            payload: data,
            oneway: false,
            request_id,
        };

        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Actor request failed: {}",
                response.status()
            ));
        }

        let actor_response: ActorResponse = response.json().await?;
        let response_payload = actor_response
            .result
            .map_err(|e| anyhow::anyhow!("Actor error: {}", e))?;

        Ok(Message::single("", response_payload))
    }

    /// Send a one-way message to an actor (fire-and-forget)
    pub async fn send_oneway(
        &self,
        addr: SocketAddr,
        actor_name: &str,
        msg: Message,
    ) -> anyhow::Result<()> {
        let request_id = self.next_request_id();
        let url = format!("http://{}/actors/{}", addr, actor_name);

        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!(
                "Streaming not supported by HTTP/1.1 transport"
            ));
        };
        let request = ActorRequest {
            msg_type,
            payload: data,
            oneway: true,
            request_id,
        };

        let response = self.client.put(&url).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Actor tell failed: {}", response.status()));
        }

        Ok(())
    }

    /// Send a request to a named actor and wait for response
    pub async fn send_named_request(
        &self,
        addr: SocketAddr,
        path: &ActorPath,
        msg: Message,
    ) -> anyhow::Result<Message> {
        let request_id = self.next_request_id();
        let url = format!("http://{}/named/{}", addr, path.as_str());

        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!(
                "Streaming not supported by HTTP/1.1 transport"
            ));
        };
        let request = ActorRequest {
            msg_type,
            payload: data,
            oneway: false,
            request_id,
        };

        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Named actor request failed: {}",
                response.status()
            ));
        }

        let actor_response: ActorResponse = response.json().await?;
        let response_payload = actor_response
            .result
            .map_err(|e| anyhow::anyhow!("Named actor error: {}", e))?;

        Ok(Message::single("", response_payload))
    }

    /// Send a one-way message to a named actor (fire-and-forget)
    pub async fn send_named_oneway(
        &self,
        addr: SocketAddr,
        path: &ActorPath,
        msg: Message,
    ) -> anyhow::Result<()> {
        let request_id = self.next_request_id();
        let url = format!("http://{}/named/{}", addr, path.as_str());

        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!(
                "Streaming not supported by HTTP/1.1 transport"
            ));
        };
        let request = ActorRequest {
            msg_type,
            payload: data,
            oneway: true,
            request_id,
        };

        let response = self.client.put(&url).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Named actor tell failed: {}",
                response.status()
            ));
        }

        Ok(())
    }

    /// Send a gossip message
    pub async fn send_gossip(
        &self,
        addr: SocketAddr,
        payload: Vec<u8>,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let url = format!("http://{}/cluster/gossip", addr);

        let request = GossipRequest { payload };

        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Gossip request failed: {}",
                response.status()
            ));
        }

        let gossip_response: GossipResponse = response.json().await?;
        Ok(gossip_response.payload)
    }
}

/// Target type for remote transport
#[derive(Clone)]
enum RemoteTarget {
    /// Target by actor name
    Actor(String),
    /// Target by named actor path
    Named(ActorPath),
}

/// Remote transport for ActorRef
pub struct HttpRemoteTransport {
    transport: Arc<HttpTransport>,
    remote_addr: SocketAddr,
    target: RemoteTarget,
}

impl HttpRemoteTransport {
    /// Create a new remote transport targeting an actor by name
    pub fn new(transport: Arc<HttpTransport>, remote_addr: SocketAddr, actor_name: String) -> Self {
        Self {
            transport,
            remote_addr,
            target: RemoteTarget::Actor(actor_name),
        }
    }

    /// Create a new remote transport targeting a named actor by path
    pub fn new_named(
        transport: Arc<HttpTransport>,
        remote_addr: SocketAddr,
        path: ActorPath,
    ) -> Self {
        Self {
            transport,
            remote_addr,
            target: RemoteTarget::Named(path),
        }
    }
}

#[async_trait::async_trait]
impl crate::actor::RemoteTransport for HttpRemoteTransport {
    async fn request(
        &self,
        _actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        let msg = Message::single(msg_type, payload);

        let response = match &self.target {
            RemoteTarget::Actor(name) => {
                self.transport
                    .send_request(self.remote_addr, name, msg)
                    .await?
            }
            RemoteTarget::Named(path) => {
                self.transport
                    .send_named_request(self.remote_addr, path, msg)
                    .await?
            }
        };

        let Message::Single { data, .. } = response else {
            return Err(anyhow::anyhow!("Unexpected stream response"));
        };
        Ok(data)
    }

    async fn send(
        &self,
        _actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        let msg = Message::single(msg_type, payload);

        match &self.target {
            RemoteTarget::Actor(name) => {
                self.transport
                    .send_oneway(self.remote_addr, name, msg)
                    .await
            }
            RemoteTarget::Named(path) => {
                self.transport
                    .send_named_oneway(self.remote_addr, path, msg)
                    .await
            }
        }
    }

    async fn request_stream(
        &self,
        _actor_id: &ActorId,
        _msg_type: &str,
        _payload: Vec<u8>,
    ) -> anyhow::Result<PayloadStream> {
        // HTTP/1.1 transport does not support streaming.
        // Use HTTP/2 transport for streaming support.
        Err(anyhow::anyhow!(
            "Streaming not supported with HTTP/1.1 transport. Use HTTP/2 transport instead."
        ))
    }
}

// ============================================================================
// HTTP Handlers
// ============================================================================

async fn health_handler(State(state): State<Arc<HttpState>>) -> impl IntoResponse {
    Json(state.handler.get_node_info().await)
}

/// POST /actors/{name} - Ask pattern: request-response
async fn actor_handler(
    State(state): State<Arc<HttpState>>,
    Path(name): Path<String>,
    Json(request): Json<ActorRequest>,
) -> impl IntoResponse {
    let msg = Message::single(request.msg_type, request.payload);

    match state.handler.handle_actor_message(&name, msg).await {
        Ok(response) => {
            let payload = match response {
                Message::Single { data, .. } => data,
                Message::Stream { .. } => Vec::new(),
            };
            (
                StatusCode::OK,
                Json(ActorResponse {
                    request_id: request.request_id,
                    result: Ok(payload),
                }),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ActorResponse {
                request_id: request.request_id,
                result: Err(e.to_string()),
            }),
        ),
    }
}

/// PUT /actors/{name} - Tell pattern: fire-and-forget
async fn actor_tell_handler(
    State(state): State<Arc<HttpState>>,
    Path(name): Path<String>,
    Json(request): Json<ActorRequest>,
) -> impl IntoResponse {
    let msg = Message::single(request.msg_type, request.payload);

    match state.handler.handle_actor_tell(&name, msg).await {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn actor_info_handler(
    State(state): State<Arc<HttpState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.handler.get_actor_info(&name).await {
        Some(info) => (StatusCode::OK, Json(info)),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Actor not found"})),
        ),
    }
}

/// POST /named/{path} - Ask pattern: request-response
async fn named_actor_handler(
    State(state): State<Arc<HttpState>>,
    Path(path): Path<String>,
    Json(request): Json<ActorRequest>,
) -> impl IntoResponse {
    let msg = Message::single(request.msg_type, request.payload);

    match state.handler.handle_named_actor_message(&path, msg).await {
        Ok(response) => {
            let payload = match response {
                Message::Single { data, .. } => data,
                Message::Stream { .. } => Vec::new(),
            };
            (
                StatusCode::OK,
                Json(ActorResponse {
                    request_id: request.request_id,
                    result: Ok(payload),
                }),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ActorResponse {
                request_id: request.request_id,
                result: Err(e.to_string()),
            }),
        ),
    }
}

/// PUT /named/{path} - Tell pattern: fire-and-forget
async fn named_actor_tell_handler(
    State(state): State<Arc<HttpState>>,
    Path(path): Path<String>,
    Json(request): Json<ActorRequest>,
) -> impl IntoResponse {
    let msg = Message::single(request.msg_type, request.payload);

    match state.handler.handle_named_actor_tell(&path, msg).await {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn named_actor_info_handler(
    State(state): State<Arc<HttpState>>,
    Path(path): Path<String>,
) -> impl IntoResponse {
    match state.handler.get_named_actor_info(&path).await {
        Some(info) => (StatusCode::OK, Json(info)),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Named actor not found", "path": path})),
        ),
    }
}

async fn gossip_handler(
    State(state): State<Arc<HttpState>>,
    Json(request): Json<GossipRequest>,
) -> impl IntoResponse {
    match state.handler.handle_gossip_message(request.payload).await {
        Ok(response_payload) => (
            StatusCode::OK,
            Json(GossipResponse {
                payload: response_payload,
            }),
        ),
        Err(e) => {
            tracing::warn!(error = %e, "Gossip handler error");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(GossipResponse { payload: None }),
            )
        }
    }
}

async fn registry_handler(State(state): State<Arc<HttpState>>) -> impl IntoResponse {
    let info = state.handler.get_node_info().await;
    // Extract just the cluster registry info
    let registry = info
        .get("cluster")
        .cloned()
        .unwrap_or(serde_json::json!(null));
    Json(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_config_default() {
        let config = HttpTransportConfig::default();
        assert_eq!(config.request_timeout, Duration::from_secs(30));
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_actor_request_serialization() {
        let request = ActorRequest {
            msg_type: "test".to_string(),
            payload: vec![1, 2, 3],
            oneway: false,
            request_id: 42,
        };

        let json = serde_json::to_string(&request).unwrap();
        let decoded: ActorRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.msg_type, "test");
        assert_eq!(decoded.payload, vec![1, 2, 3]);
        assert_eq!(decoded.request_id, 42);
    }

    #[test]
    fn test_gossip_request_serialization() {
        let request = GossipRequest {
            payload: vec![1, 2, 3, 4],
        };

        let json = serde_json::to_string(&request).unwrap();
        let decoded: GossipRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.payload, vec![1, 2, 3, 4]);
    }
}
