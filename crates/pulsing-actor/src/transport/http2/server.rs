//! HTTP/2 Server implementation
//!
//! Supports h2c (HTTP/2 over cleartext) with optional HTTP/1.1 fallback.
//! When `tls` feature is enabled, supports TLS with passphrase-derived certificates.

use super::config::Http2Config;
use super::stream::{BinaryFrameParser, StreamFrame};
use super::{headers, MessageMode, RequestType};
use crate::actor::Message;
use crate::tracing::{TraceContext, TRACEPARENT_HEADER};
use bytes::Bytes;
use futures::StreamExt;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::server::conn::http2;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Handler trait for HTTP/2 server
#[async_trait::async_trait]
pub trait Http2ServerHandler: Send + Sync + 'static {
    /// Unified message handler - accepts and returns Message (Single or Stream)
    ///
    /// This is the primary method that should be implemented. It handles both:
    /// - Single requests: `Message::Single` with payload
    /// - Streaming requests: `Message::Stream` with async stream of chunks
    ///
    /// The default implementation delegates to `handle_message_simple` for backward compatibility.
    async fn handle_message_full(&self, path: &str, msg: Message) -> anyhow::Result<Message> {
        // Default: extract single message and delegate to simple handler
        match msg {
            Message::Single { msg_type, data } => {
                self.handle_message_simple(path, &msg_type, data).await
            }
            Message::Stream { .. } => Err(anyhow::anyhow!(
                "Streaming requests not supported by this handler"
            )),
        }
    }

    /// Simple message handler for backward compatibility
    ///
    /// Implement this if you only need to handle single (non-streaming) requests.
    async fn handle_message_simple(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Message> {
        let _ = (path, msg_type, payload);
        Err(anyhow::anyhow!("Not implemented"))
    }

    /// Handle tell (fire-and-forget) message
    async fn handle_tell(&self, path: &str, msg_type: &str, payload: Vec<u8>)
        -> anyhow::Result<()>;

    /// Handle gossip message
    async fn handle_gossip(
        &self,
        payload: Vec<u8>,
        peer_addr: SocketAddr,
    ) -> anyhow::Result<Option<Vec<u8>>>;

    /// Get health status
    async fn health_check(&self) -> serde_json::Value {
        serde_json::json!({"status": "ok"})
    }

    /// Get Prometheus metrics (text format)
    ///
    /// Default implementation returns empty metrics.
    /// Override this to provide system metrics.
    async fn prometheus_metrics(&self) -> String {
        String::new()
    }

    /// Get cluster members list (for CLI tools)
    ///
    /// Returns JSON array of member information.
    async fn cluster_members(&self) -> serde_json::Value {
        serde_json::json!([])
    }

    /// Get actors list on this node (for CLI tools)
    ///
    /// Returns JSON array of actor information.
    async fn actors_list(&self, include_internal: bool) -> serde_json::Value {
        let _ = include_internal;
        serde_json::json!([])
    }
}

/// HTTP/2 Server
pub struct Http2Server {
    local_addr: SocketAddr,
    cancel: CancellationToken,
}

impl Http2Server {
    /// Create and start a new HTTP/2 server
    pub async fn new(
        bind_addr: SocketAddr,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(bind_addr).await?;
        let local_addr = listener.local_addr()?;

        tracing::info!(addr = %local_addr, "Starting HTTP/2 server");

        // Spawn the server task
        let server_cancel = cancel.clone();
        tokio::spawn(async move {
            Self::run_server(listener, handler, config, server_cancel).await;
        });

        Ok(Self { local_addr, cancel })
    }

    /// Get the local address the server is bound to
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Shutdown the server
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    /// Run the server loop
    async fn run_server(
        listener: TcpListener,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) {
        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, peer_addr)) => {
                            let handler = handler.clone();
                            let config = config.clone();
                            let conn_cancel = cancel.clone();

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(
                                    stream,
                                    peer_addr,
                                    handler,
                                    config,
                                    conn_cancel,
                                ).await {
                                    tracing::debug!(
                                        peer = %peer_addr,
                                        error = %e,
                                        "Connection error"
                                    );
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Accept error");
                        }
                    }
                }
                _ = cancel.cancelled() => {
                    tracing::info!("HTTP/2 server shutting down");
                    break;
                }
            }
        }
    }

    /// Handle a single connection
    async fn handle_connection(
        stream: tokio::net::TcpStream,
        peer_addr: SocketAddr,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        // Check if TLS is enabled
        #[cfg(feature = "tls")]
        if let Some(ref tls_config) = config.tls {
            // TLS mode: accept TLS handshake first
            let tls_stream = tls_config.accept(stream).await?;
            let io = TokioIo::new(tls_stream);

            // TLS connections always use HTTP/2 (no HTTP/1.1 fallback)
            return Self::serve_h2_generic(io, peer_addr, handler, config, cancel).await;
        }

        // Plain TCP mode (h2c) - HTTP/2 only (prior knowledge)
        let io = TokioIo::new(stream);
        Self::serve_h2(io, peer_addr, handler, config, cancel).await
    }

    /// Serve HTTP/2 only (prior knowledge mode)
    async fn serve_h2(
        io: TokioIo<tokio::net::TcpStream>,
        peer_addr: SocketAddr,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        Self::serve_h2_generic(io, peer_addr, handler, config, cancel).await
    }

    /// Generic HTTP/2 server - works with any IO type (TCP or TLS)
    async fn serve_h2_generic<I>(
        io: TokioIo<I>,
        peer_addr: SocketAddr,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) -> anyhow::Result<()>
    where
        I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let service = service_fn(move |req| {
            let handler = handler.clone();
            async move { Self::handle_request(req, handler, peer_addr).await }
        });

        let mut h2_builder = http2::Builder::new(TokioExecutor::new());
        h2_builder
            .max_concurrent_streams(config.max_concurrent_streams)
            .initial_stream_window_size(config.initial_window_size)
            .initial_connection_window_size(config.initial_connection_window_size)
            .max_frame_size(config.max_frame_size)
            .max_header_list_size(config.max_header_list_size);

        let conn = h2_builder.serve_connection(io, service);

        tokio::select! {
            result = conn => {
                if let Err(e) = result {
                    tracing::debug!(peer = %peer_addr, error = %e, "HTTP/2 connection error");
                }
            }
            _ = cancel.cancelled() => {
                tracing::debug!(peer = %peer_addr, "Connection cancelled");
            }
        }

        Ok(())
    }

    /// Handle a single HTTP request
    async fn handle_request(
        req: Request<Incoming>,
        handler: Arc<dyn Http2ServerHandler>,
        peer_addr: SocketAddr,
    ) -> Result<Response<BoxBody>, Infallible> {
        let path = req.uri().path().to_string();
        let method = req.method().clone();

        // Extract trace context from incoming request
        let parent_trace = req
            .headers()
            .get(TRACEPARENT_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(TraceContext::from_traceparent);

        // Create a child span ID for this request
        let trace_ctx = parent_trace.map(|p| p.child()).unwrap_or_default();

        // Create span with trace context
        let span = tracing::info_span!(
            "http.request",
            otel.name = %format!("{} {}", method, path),
            http.method = %method,
            http.route = %path,
            http.peer = %peer_addr,
            trace_id = %trace_ctx.trace_id,
            span_id = %trace_ctx.span_id,
        );

        // Enter the span for the duration of request handling
        let _enter = span.enter();

        // Health check endpoint
        if path == "/health" && method == Method::GET {
            let health = handler.health_check().await;
            let body = serde_json::to_vec(&health).unwrap_or_default();
            return Ok(json_response(StatusCode::OK, body));
        }

        // Prometheus metrics endpoint
        if path == "/metrics" && method == Method::GET {
            let metrics = handler.prometheus_metrics().await;
            return Ok(text_response(StatusCode::OK, metrics.into_bytes()));
        }

        // Cluster members endpoint (for CLI tools)
        if path == "/cluster/members" && method == Method::GET {
            let members = handler.cluster_members().await;
            let body = serde_json::to_vec(&members).unwrap_or_default();
            return Ok(json_response(StatusCode::OK, body));
        }

        // Actors list endpoint (for CLI tools)
        if (path == "/actors" || path == "/actors/") && method == Method::GET {
            // Check for ?all=true query parameter
            let include_internal = req
                .uri()
                .query()
                .map(|q| q.contains("all=true"))
                .unwrap_or(false);
            let actors = handler.actors_list(include_internal).await;
            let body = serde_json::to_vec(&actors).unwrap_or_default();
            return Ok(json_response(StatusCode::OK, body));
        }

        // Only POST for actor messages
        if method != Method::POST {
            return Ok(error_response(
                StatusCode::METHOD_NOT_ALLOWED,
                b"Method not allowed".to_vec(),
            ));
        }

        // Extract headers
        let mode = req
            .headers()
            .get(headers::MESSAGE_MODE)
            .and_then(|v| v.to_str().ok())
            .and_then(MessageMode::parse)
            .unwrap_or(MessageMode::Ask);

        let msg_type = req
            .headers()
            .get(headers::MESSAGE_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        // Check if this is a streaming request
        let request_type = req
            .headers()
            .get(headers::REQUEST_TYPE)
            .and_then(|v| v.to_str().ok())
            .and_then(RequestType::parse)
            .unwrap_or(RequestType::Single);

        // Dispatch based on mode
        if path == "/cluster/gossip" {
            // Gossip doesn't support streaming
            let body_bytes = match req.collect().await {
                Ok(collected) => collected.to_bytes().to_vec(),
                Err(e) => {
                    return Ok(error_response(
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read body: {}", e).into_bytes(),
                    ));
                }
            };
            return Self::handle_gossip_request(&handler, body_bytes, peer_addr).await;
        }

        match mode {
            MessageMode::Tell => {
                // Tell doesn't support streaming requests
                let body_bytes = match req.collect().await {
                    Ok(collected) => collected.to_bytes().to_vec(),
                    Err(e) => {
                        return Ok(error_response(
                            StatusCode::BAD_REQUEST,
                            format!("Failed to read body: {}", e).into_bytes(),
                        ));
                    }
                };
                Self::handle_tell_request(&handler, &path, &msg_type, body_bytes).await
            }
            // Ask and Stream mode - check for streaming request
            MessageMode::Ask | MessageMode::Stream => {
                match request_type {
                    RequestType::Single => {
                        // Single request - read body as bytes
                        let body_bytes = match req.collect().await {
                            Ok(collected) => collected.to_bytes().to_vec(),
                            Err(e) => {
                                return Ok(error_response(
                                    StatusCode::BAD_REQUEST,
                                    format!("Failed to read body: {}", e).into_bytes(),
                                ));
                            }
                        };
                        let msg = Message::single(&msg_type, body_bytes);
                        Self::handle_message_request_full(&handler, &path, msg).await
                    }
                    RequestType::Stream => {
                        // Streaming request - parse binary frames into Message::Stream
                        let msg = Self::parse_streaming_request(req.into_body(), &msg_type);
                        Self::handle_message_request_full(&handler, &path, msg).await
                    }
                }
            }
        }
    }

    /// Parse a streaming request body (binary frames) into Message::Stream
    fn parse_streaming_request(body: Incoming, default_msg_type: &str) -> Message {
        let (tx, rx) = mpsc::channel::<anyhow::Result<Message>>(32);
        let default_msg_type = default_msg_type.to_string();

        // Spawn task to parse binary frames
        tokio::spawn(async move {
            let mut parser = BinaryFrameParser::new();
            let mut body_stream = http_body_util::BodyStream::new(body);

            while let Some(result) = body_stream.next().await {
                match result {
                    Ok(frame) => {
                        if let Ok(data) = frame.into_data() {
                            parser.push(&data);

                            // Parse all complete frames
                            for frame_result in parser.parse_all() {
                                match frame_result {
                                    Ok(frame) => {
                                        // Skip end frames
                                        if frame.end && frame.get_data().is_empty() {
                                            continue;
                                        }
                                        // Convert frame to message
                                        match frame.to_message() {
                                            Ok(Some(msg)) => {
                                                if tx.send(Ok(msg)).await.is_err() {
                                                    return;
                                                }
                                            }
                                            Ok(None) => {} // Skip empty frames
                                            Err(e) => {
                                                let _ = tx.send(Err(e)).await;
                                                return;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx.send(Err(e)).await;
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(Err(anyhow::anyhow!("Body read error: {}", e)))
                            .await;
                        return;
                    }
                }
            }
        });

        // Create Message::Stream from channel
        Message::from_channel(default_msg_type, rx)
    }

    async fn handle_gossip_request(
        handler: &Arc<dyn Http2ServerHandler>,
        payload: Vec<u8>,
        peer_addr: SocketAddr,
    ) -> Result<Response<BoxBody>, Infallible> {
        match handler.handle_gossip(payload, peer_addr).await {
            Ok(Some(response)) => Ok(octet_response(StatusCode::OK, response)),
            Ok(None) => Ok(empty_response(StatusCode::OK)),
            Err(e) => Ok(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string().into_bytes(),
            )),
        }
    }

    /// Unified message handler - accepts Message (Single or Stream) and handles response
    async fn handle_message_request_full(
        handler: &Arc<dyn Http2ServerHandler>,
        path: &str,
        msg: Message,
    ) -> Result<Response<BoxBody>, Infallible> {
        match handler.handle_message_full(path, msg).await {
            Ok(Message::Single { data, .. }) => {
                // Single response - return directly with response type header
                Ok(octet_response_with_headers(
                    StatusCode::OK,
                    full_body(data),
                    &[(headers::RESPONSE_TYPE, "single")],
                ))
            }
            Ok(Message::Stream {
                default_msg_type,
                stream,
            }) => {
                // Stream response - convert Message stream to binary frames
                let (tx, rx) = mpsc::channel::<Result<Frame<Bytes>, Infallible>>(32);

                tokio::spawn(async move {
                    let mut stream = std::pin::pin!(stream);

                    while let Some(result) = stream.next().await {
                        let frame = match result {
                            Ok(msg) => StreamFrame::from_message(&msg, &default_msg_type),
                            Err(e) => StreamFrame::error(e.to_string()),
                        };

                        if tx.send(Ok(Frame::data(frame.to_binary()))).await.is_err() {
                            break;
                        }
                    }

                    // Send end frame
                    let _ = tx
                        .send(Ok(Frame::data(StreamFrame::end().to_binary())))
                        .await;
                });

                let body_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                let body = StreamBody::new(body_stream);

                Ok(octet_response_with_headers(
                    StatusCode::OK,
                    BoxBody::new(body),
                    &[(headers::RESPONSE_TYPE, "stream")],
                ))
            }
            Err(e) => Ok(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string().into_bytes(),
            )),
        }
    }

    async fn handle_tell_request(
        handler: &Arc<dyn Http2ServerHandler>,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<Response<BoxBody>, Infallible> {
        match handler.handle_tell(path, msg_type, payload).await {
            Ok(()) => Ok(empty_response(StatusCode::ACCEPTED)),
            Err(e) => Ok(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string().into_bytes(),
            )),
        }
    }
}

// Body type aliases
type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

fn full_body(data: Vec<u8>) -> BoxBody {
    BoxBody::new(Full::new(Bytes::from(data)).map_err(|_| unreachable!()))
}

fn empty_body() -> BoxBody {
    BoxBody::new(http_body_util::Empty::new().map_err(|_| unreachable!()))
}

// ============================================================================
// Response builder helpers - eliminates unwrap() calls with proper error handling
// ============================================================================

/// Build a JSON response with the given status code
fn json_response(status: StatusCode, body: Vec<u8>) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(full_body(body))
        .expect("failed to build JSON response with valid headers")
}

/// Build an octet-stream response with the given status code
fn octet_response(status: StatusCode, body: Vec<u8>) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .header("content-type", "application/octet-stream")
        .body(full_body(body))
        .expect("failed to build octet-stream response with valid headers")
}

/// Build an octet-stream response with custom headers
fn octet_response_with_headers(
    status: StatusCode,
    body: BoxBody,
    headers: &[(&str, &str)],
) -> Response<BoxBody> {
    let mut builder = Response::builder()
        .status(status)
        .header("content-type", "application/octet-stream");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder
        .body(body)
        .expect("failed to build response with valid headers")
}

/// Build a text response (for metrics endpoint)
fn text_response(status: StatusCode, body: Vec<u8>) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .header("content-type", "text/plain; version=0.0.4; charset=utf-8")
        .body(full_body(body))
        .expect("failed to build text response with valid headers")
}

/// Build an empty response with the given status code
fn empty_response(status: StatusCode) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .body(empty_body())
        .expect("failed to build empty response")
}

/// Build an error response with the given status code and message
fn error_response(status: StatusCode, message: impl Into<Vec<u8>>) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .body(full_body(message.into()))
        .expect("failed to build error response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Message;

    struct MockHandler;

    #[async_trait::async_trait]
    impl Http2ServerHandler for MockHandler {
        async fn handle_message_simple(
            &self,
            _path: &str,
            _msg_type: &str,
            payload: Vec<u8>,
        ) -> anyhow::Result<Message> {
            // Echo the payload as single message
            Ok(Message::single("", payload))
        }

        async fn handle_tell(
            &self,
            _path: &str,
            _msg_type: &str,
            _payload: Vec<u8>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn handle_gossip(
            &self,
            _payload: Vec<u8>,
            _peer_addr: SocketAddr,
        ) -> anyhow::Result<Option<Vec<u8>>> {
            Ok(None)
        }
    }

    #[tokio::test]
    async fn test_server_creation() {
        let handler = Arc::new(MockHandler);
        let cancel = CancellationToken::new();

        let server = Http2Server::new(
            "127.0.0.1:0".parse().unwrap(),
            handler,
            Http2Config::default(),
            cancel.clone(),
        )
        .await
        .unwrap();

        assert_ne!(server.local_addr().port(), 0);
        cancel.cancel();
    }
}
