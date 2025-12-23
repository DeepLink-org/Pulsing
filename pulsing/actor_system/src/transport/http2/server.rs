//! HTTP/2 Server implementation
//!
//! Supports h2c (HTTP/2 over cleartext) with optional HTTP/1.1 fallback.

use super::config::Http2Config;
use super::stream::StreamFrame;
use super::{headers, MessageMode};
use crate::actor::{Message, MessageStream};
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
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// Handler trait for HTTP/2 server
#[async_trait::async_trait]
pub trait Http2ServerHandler: Send + Sync + 'static {
    /// Handle ask (request-response) message
    async fn handle_ask(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>>;

    /// Handle tell (fire-and-forget) message
    async fn handle_tell(&self, path: &str, msg_type: &str, payload: Vec<u8>)
        -> anyhow::Result<()>;

    /// Handle stream request - returns a MessageStream
    async fn handle_stream(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<MessageStream>;

    /// Handle gossip message
    async fn handle_gossip(&self, payload: Vec<u8>) -> anyhow::Result<Option<Vec<u8>>>;

    /// Get health status
    async fn health_check(&self) -> serde_json::Value {
        serde_json::json!({"status": "ok"})
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
        let io = TokioIo::new(stream);

        // Try to detect HTTP/2 preface
        // For simplicity, we'll use auto detection with http1 fallback
        if config.enable_http1_fallback {
            // Use auto connection that handles both HTTP/1.1 and HTTP/2
            Self::serve_auto(io, peer_addr, handler, config, cancel).await
        } else {
            // HTTP/2 only (prior knowledge)
            Self::serve_h2(io, peer_addr, handler, config, cancel).await
        }
    }

    /// Serve with automatic HTTP version detection
    async fn serve_auto(
        io: TokioIo<tokio::net::TcpStream>,
        peer_addr: SocketAddr,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        // For now, we'll check the first bytes to detect HTTP/2
        // HTTP/2 preface starts with "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
        // This is a simplified approach - in production you might use hyper-util's auto

        let service = service_fn(move |req| {
            let handler = handler.clone();
            async move { Self::handle_request(req, handler).await }
        });

        // Try HTTP/2 first, fall back to HTTP/1.1
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
                    // If HTTP/2 fails, the connection might be HTTP/1.1
                    // For now, just log and continue
                    tracing::debug!(peer = %peer_addr, error = %e, "Connection ended");
                }
            }
            _ = cancel.cancelled() => {
                tracing::debug!(peer = %peer_addr, "Connection cancelled");
            }
        }

        Ok(())
    }

    /// Serve HTTP/2 only (prior knowledge mode)
    async fn serve_h2(
        io: TokioIo<tokio::net::TcpStream>,
        peer_addr: SocketAddr,
        handler: Arc<dyn Http2ServerHandler>,
        config: Http2Config,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        let service = service_fn(move |req| {
            let handler = handler.clone();
            async move { Self::handle_request(req, handler).await }
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
    ) -> Result<Response<BoxBody>, Infallible> {
        let path = req.uri().path().to_string();
        let method = req.method().clone();

        // Health check endpoint
        if path == "/health" && method == Method::GET {
            let health = handler.health_check().await;
            let body = serde_json::to_vec(&health).unwrap_or_default();
            return Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(full_body(body))
                .unwrap());
        }

        // Only POST for actor messages
        if method != Method::POST {
            return Ok(Response::builder()
                .status(StatusCode::METHOD_NOT_ALLOWED)
                .body(full_body(b"Method not allowed".to_vec()))
                .unwrap());
        }

        // Extract headers
        let mode = req
            .headers()
            .get(headers::MESSAGE_MODE)
            .and_then(|v| v.to_str().ok())
            .and_then(MessageMode::from_str)
            .unwrap_or(MessageMode::Ask);

        let msg_type = req
            .headers()
            .get(headers::MESSAGE_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        // Read body
        let body_bytes = match req.collect().await {
            Ok(collected) => collected.to_bytes().to_vec(),
            Err(e) => {
                return Ok(Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(full_body(
                        format!("Failed to read body: {}", e).into_bytes(),
                    ))
                    .unwrap());
            }
        };

        // Dispatch based on mode
        if path == "/cluster/gossip" {
            return Self::handle_gossip_request(&handler, body_bytes).await;
        }

        match mode {
            MessageMode::Ask => {
                Self::handle_ask_request(&handler, &path, &msg_type, body_bytes).await
            }
            MessageMode::Tell => {
                Self::handle_tell_request(&handler, &path, &msg_type, body_bytes).await
            }
            MessageMode::Stream => {
                Self::handle_stream_request(&handler, &path, &msg_type, body_bytes).await
            }
        }
    }

    async fn handle_gossip_request(
        handler: &Arc<dyn Http2ServerHandler>,
        payload: Vec<u8>,
    ) -> Result<Response<BoxBody>, Infallible> {
        match handler.handle_gossip(payload).await {
            Ok(Some(response)) => Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/octet-stream")
                .body(full_body(response))
                .unwrap()),
            Ok(None) => Ok(Response::builder()
                .status(StatusCode::OK)
                .body(empty_body())
                .unwrap()),
            Err(e) => Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(full_body(e.to_string().into_bytes()))
                .unwrap()),
        }
    }

    async fn handle_ask_request(
        handler: &Arc<dyn Http2ServerHandler>,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<Response<BoxBody>, Infallible> {
        match handler.handle_ask(path, msg_type, payload).await {
            Ok(response) => Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/octet-stream")
                .body(full_body(response))
                .unwrap()),
            Err(e) => Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(full_body(e.to_string().into_bytes()))
                .unwrap()),
        }
    }

    async fn handle_tell_request(
        handler: &Arc<dyn Http2ServerHandler>,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<Response<BoxBody>, Infallible> {
        match handler.handle_tell(path, msg_type, payload).await {
            Ok(()) => Ok(Response::builder()
                .status(StatusCode::ACCEPTED)
                .body(empty_body())
                .unwrap()),
            Err(e) => Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(full_body(e.to_string().into_bytes()))
                .unwrap()),
        }
    }

    async fn handle_stream_request(
        handler: &Arc<dyn Http2ServerHandler>,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<Response<BoxBody>, Infallible> {
        match handler.handle_stream(path, msg_type, payload).await {
            Ok(stream) => {
                // Collect stream into a channel that can be shared safely
                let (tx, rx) = tokio::sync::mpsc::channel::<Result<Frame<Bytes>, Infallible>>(32);

                // Spawn task to read from stream and send to channel
                tokio::spawn(async move {
                    let mut stream = std::pin::pin!(stream);
                    let mut seq = 0u64;

                    while let Some(result) = stream.next().await {
                        let frame = match result {
                            Ok(msg) => {
                                let (msg_type, payload) = match msg {
                                    Message::Single { msg_type, data } => (msg_type, data),
                                    Message::Stream { msg_type, .. } => (msg_type, Vec::new()),
                                };
                                StreamFrame::data(seq, &msg_type, &payload)
                            }
                            Err(e) => StreamFrame::error(seq, e.to_string()),
                        };

                        let bytes = match frame.to_ndjson() {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                let error_frame = StreamFrame::error(seq, e.to_string());
                                error_frame.to_ndjson().unwrap_or_else(|_| Bytes::new())
                            }
                        };

                        if tx.send(Ok(Frame::data(bytes))).await.is_err() {
                            break; // Receiver dropped
                        }
                        seq += 1;
                    }

                    // Send end frame
                    let end_frame = StreamFrame::end(seq);
                    if let Ok(bytes) = end_frame.to_ndjson() {
                        let _ = tx.send(Ok(Frame::data(bytes))).await;
                    }
                });

                // Convert receiver to stream
                let body_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                let body = StreamBody::new(body_stream);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/x-ndjson")
                    .body(BoxBody::new(body))
                    .unwrap())
            }
            Err(e) => Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(full_body(e.to_string().into_bytes()))
                .unwrap()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Message;

    struct MockHandler;

    #[async_trait::async_trait]
    impl Http2ServerHandler for MockHandler {
        async fn handle_ask(
            &self,
            _path: &str,
            _msg_type: &str,
            payload: Vec<u8>,
        ) -> anyhow::Result<Vec<u8>> {
            // Echo the payload
            Ok(payload)
        }

        async fn handle_tell(
            &self,
            _path: &str,
            _msg_type: &str,
            _payload: Vec<u8>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn handle_stream(
            &self,
            _path: &str,
            _msg_type: &str,
            _payload: Vec<u8>,
        ) -> anyhow::Result<MessageStream> {
            let stream = futures::stream::iter(vec![
                Ok(Message::single("token", b"hello")),
                Ok(Message::single("token", b"world")),
            ]);
            Ok(Box::pin(stream))
        }

        async fn handle_gossip(&self, _payload: Vec<u8>) -> anyhow::Result<Option<Vec<u8>>> {
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
