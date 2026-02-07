//! HTTP/2 client implementation.

use super::config::Http2Config;
use super::pool::{ConnectionPool, PoolConfig};
use super::retry::{RetryConfig, RetryExecutor};
use super::stream::{BinaryFrameParser, StreamFrame, StreamHandle};
use super::{headers, MessageMode, RequestType};
use crate::actor::{Message, MessageStream};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::tracing::{TraceContext, TRACEPARENT_HEADER};
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Method, Request};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Context for fault injection (testing / chaos).
#[derive(Clone, Debug)]
pub struct FaultInjectContext {
    pub addr: SocketAddr,
    pub path: String,
    pub msg_type: String,
    pub operation: FaultInjectOperation,
}

/// Operation kind for fault injection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FaultInjectOperation {
    Ask,
    Tell,
    Stream,
}

/// Fault injector for testing: optionally return an error before performing the request.
pub trait FaultInjector: Send + Sync {
    /// If returns Some(error), the client will return this error without sending the request.
    fn inject(&self, ctx: &FaultInjectContext) -> Option<PulsingError>;
}

/// HTTP/2 client with connection pooling, retry, and timeout support.
pub struct Http2Client {
    pool: Arc<ConnectionPool>,
    config: Http2Config,
    retry_config: RetryConfig,
    cancel: CancellationToken,
    fault_injector: Option<Arc<dyn FaultInjector>>,
}

impl Http2Client {
    pub fn new(config: Http2Config) -> Self {
        Self {
            pool: Arc::new(ConnectionPool::new(config.clone())),
            config,
            retry_config: RetryConfig::default(),
            cancel: CancellationToken::new(),
            fault_injector: None,
        }
    }

    pub fn with_configs(
        http2_config: Http2Config,
        pool_config: PoolConfig,
        retry_config: RetryConfig,
    ) -> Self {
        Self {
            pool: Arc::new(ConnectionPool::with_config(
                http2_config.clone(),
                pool_config,
            )),
            config: http2_config,
            retry_config,
            cancel: CancellationToken::new(),
            fault_injector: None,
        }
    }

    pub fn with_retry(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    /// Set fault injector for testing / chaos engineering. When set, injector may return
    /// an error before the request is sent.
    pub fn with_fault_injector(self, injector: Option<Arc<dyn FaultInjector>>) -> Self {
        Self {
            fault_injector: injector,
            ..self
        }
    }

    pub fn pool(&self) -> &Arc<ConnectionPool> {
        &self.pool
    }

    pub fn stats(&self) -> &Arc<super::pool::PoolStats> {
        self.pool.stats()
    }

    pub fn start_background_tasks(&self) {
        self.pool.start_cleanup_task(self.cancel.clone());
    }

    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    fn check_fault_inject(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        op: FaultInjectOperation,
    ) -> Result<()> {
        if let Some(ref injector) = self.fault_injector {
            let ctx = FaultInjectContext {
                addr,
                path: path.to_string(),
                msg_type: msg_type.to_string(),
                operation: op,
            };
            if let Some(err) = injector.inject(&ctx) {
                return Err(err);
            }
        }
        Ok(())
    }

    pub async fn ask(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>> {
        self.check_fault_inject(addr, path, msg_type, FaultInjectOperation::Ask)?;
        let executor = RetryExecutor::new(self.retry_config.clone());
        executor
            .execute(true, || {
                self.ask_once(addr, path, msg_type, payload.clone())
            })
            .await
    }

    async fn ask_once(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>> {
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Ask)
            .await?;

        let status = response.status();

        let body = tokio::time::timeout(self.config.request_timeout, response.collect())
            .await
            .map_err(|_| {
                RuntimeError::request_timeout(self.config.request_timeout.as_millis() as u64)
            })?
            .map_err(|e| RuntimeError::Io(e.to_string()))?
            .to_bytes();

        if !status.is_success() {
            let error_msg = String::from_utf8_lossy(&body);
            return Err(PulsingError::from(RuntimeError::invalid_response(format!(
                "status {}: {}",
                status, error_msg
            ))));
        }

        Ok(body.to_vec())
    }

    pub async fn tell(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<()> {
        self.check_fault_inject(addr, path, msg_type, FaultInjectOperation::Tell)?;
        let executor = RetryExecutor::new(self.retry_config.clone());
        executor
            .execute(false, || {
                self.tell_once(addr, path, msg_type, payload.clone())
            })
            .await
    }

    async fn tell_once(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<()> {
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Tell)
            .await?;

        let status = response.status();

        if !status.is_success() {
            let body = response
                .collect()
                .await
                .map_err(|e| RuntimeError::Io(e.to_string()))?
                .to_bytes();
            let error_msg = String::from_utf8_lossy(&body);
            return Err(PulsingError::from(RuntimeError::invalid_response(format!(
                "tell status {} to {}: {}",
                status, addr, error_msg
            ))));
        }

        Ok(())
    }

    /// Send a stream request and receive streaming response as StreamFrame.
    pub async fn ask_stream(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<StreamHandle<StreamFrame>> {
        self.check_fault_inject(addr, path, msg_type, FaultInjectOperation::Stream)?;
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Stream)
            .await?;

        let status = response.status();

        if !status.is_success() {
            let body = response
                .collect()
                .await
                .map_err(|e| RuntimeError::Io(e.to_string()))?
                .to_bytes();
            let error_msg = String::from_utf8_lossy(&body);
            return Err(PulsingError::from(RuntimeError::invalid_response(format!(
                "stream status {}: {}",
                status, error_msg
            ))));
        }

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        let stream_timeout = self.config.stream_timeout;
        let body_stream = response.into_body();
        let frame_stream = Self::body_to_frame_stream(body_stream, cancel_clone, stream_timeout);

        Ok(StreamHandle::new(frame_stream, cancel))
    }

    /// Send a stream request and receive streaming response as MessageStream
    ///
    /// Each frame is converted to a `Message::Single` preserving the msg_type.
    pub async fn ask_stream_raw(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<MessageStream> {
        let stream_handle = self.ask_stream(addr, path, msg_type, payload).await?;

        let msg_stream = stream_handle.filter_map(|result| async move {
            match result {
                Ok(frame) => match frame.to_message() {
                    Ok(Some(msg)) => Some(Ok(msg)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                },
                Err(e) => Some(Err(PulsingError::from(RuntimeError::Other(e.to_string())))),
            }
        });

        Ok(Box::pin(msg_stream))
    }

    /// Unified send - automatically handles single and stream requests/responses
    /// based on the Message type and response header from server
    pub async fn send_message(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<Message> {
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Ask)
            .await?;

        self.parse_response(response).await
    }

    /// Send a Message (Single or Stream) and receive response
    ///
    /// This method supports both single and streaming requests:
    /// - `Message::Single`: Sent as a regular request body
    /// - `Message::Stream`: Sent as length-prefixed binary frames
    pub async fn send_message_full(
        &self,
        addr: SocketAddr,
        path: &str,
        msg: Message,
    ) -> Result<Message> {
        match msg {
            Message::Single { msg_type, data } => {
                self.send_message(addr, path, &msg_type, data).await
            }
            Message::Stream {
                default_msg_type,
                stream,
            } => {
                let response = self
                    .send_stream_request(addr, path, &default_msg_type, stream)
                    .await?;
                self.parse_response(response).await
            }
        }
    }

    /// Send a streaming request (Message::Stream as request body)
    ///
    /// Note: Streaming requests use a dedicated connection (not from the pool)
    /// because the connection pool is typed for `Full<Bytes>` bodies.
    async fn send_stream_request(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        stream: MessageStream,
    ) -> Result<hyper::Response<Incoming>> {
        use hyper::client::conn::http2;
        use hyper_util::rt::{TokioExecutor, TokioIo};
        use tokio::net::TcpStream;

        // Create trace context for outgoing streaming request
        let trace_ctx = TraceContext::from_current()
            .map(|p| p.child())
            .unwrap_or_default();

        let span = tracing::info_span!(
            "http.client.stream",
            otel.name = %format!("POST {} (stream)", path),
            http.method = "POST",
            http.url = %format!("http://{}{}", addr, path),
            trace_id = %trace_ctx.trace_id,
            span_id = %trace_ctx.span_id,
        );
        let _enter = span.enter();

        // Create a dedicated connection for streaming request
        let tcp_stream =
            tokio::time::timeout(self.config.connect_timeout, TcpStream::connect(addr))
                .await
                .map_err(|_| {
                    RuntimeError::connection_failed(
                        addr.to_string(),
                        "Connection timeout".to_string(),
                    )
                })?
                .map_err(|e| RuntimeError::connection_failed(addr.to_string(), e.to_string()))?;

        // Build HTTP/2 connection with streaming body type - with or without TLS
        type StreamingBody = StreamBody<
            tokio_stream::wrappers::ReceiverStream<std::result::Result<Frame<Bytes>, Infallible>>,
        >;

        #[cfg(feature = "tls")]
        if let Some(ref tls_config) = self.config.tls {
            let server_name = addr.ip().to_string();
            let tls_stream = tls_config
                .connect(tcp_stream, &server_name)
                .await
                .map_err(|e| RuntimeError::tls_error(e.to_string()))?;
            let io = TokioIo::new(tls_stream);
            let (mut sender, conn): (http2::SendRequest<StreamingBody>, _) =
                http2::handshake(TokioExecutor::new(), io)
                    .await
                    .map_err(|e| RuntimeError::tls_error(e.to_string()))?;

            // Spawn connection driver for TLS
            let cancel = self.cancel.clone();
            tokio::spawn(async move {
                tokio::select! {
                    result = conn => {
                        if let Err(e) = result {
                            tracing::debug!(error = %e, "TLS streaming connection ended");
                        }
                    }
                    _ = cancel.cancelled() => {
                        tracing::debug!("TLS streaming connection cancelled");
                    }
                }
            });

            // Complete the streaming request (TLS path)
            let (tx, rx) =
                tokio::sync::mpsc::channel::<std::result::Result<Frame<Bytes>, Infallible>>(32);
            let default_msg_type = msg_type.to_string();
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
                let _ = tx
                    .send(Ok(Frame::data(StreamFrame::end().to_binary())))
                    .await;
            });

            let body_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            let body = StreamBody::new(body_stream);

            let uri = format!("http://{}{}", addr, path);
            let request = Request::builder()
                .method(Method::POST)
                .uri(&uri)
                .header(headers::MESSAGE_MODE, MessageMode::Ask.as_str())
                .header(headers::MESSAGE_TYPE, msg_type)
                .header(headers::REQUEST_TYPE, RequestType::Stream.as_str())
                .header(TRACEPARENT_HEADER, trace_ctx.to_traceparent())
                .header("content-type", "application/octet-stream")
                .body(body)
                .map_err(|e| {
                    RuntimeError::protocol_error(format!("Failed to build request: {}", e))
                })?;

            let send_future = sender.send_request(request);
            let response = tokio::time::timeout(self.config.stream_timeout, send_future)
                .await
                .map_err(|_| {
                    RuntimeError::request_timeout(self.config.stream_timeout.as_millis() as u64)
                })?
                .map_err(|e| RuntimeError::protocol_error(e.to_string()))?;

            return Ok(response);
        }

        // Plain h2c mode (no TLS or TLS feature disabled)
        let io = TokioIo::new(tcp_stream);
        let (mut sender, conn): (http2::SendRequest<StreamingBody>, _) =
            http2::handshake(TokioExecutor::new(), io)
                .await
                .map_err(|e| {
                    RuntimeError::protocol_error(format!("HTTP/2 handshake failed: {}", e))
                })?;

        // Spawn connection driver
        let cancel = self.cancel.clone();
        tokio::spawn(async move {
            tokio::select! {
                result = conn => {
                    if let Err(e) = result {
                        tracing::debug!(error = %e, "Streaming connection ended");
                    }
                }
                _ = cancel.cancelled() => {
                    tracing::debug!("Streaming connection cancelled");
                }
            }
        });

        // Create a channel for streaming body
        let (tx, rx) =
            tokio::sync::mpsc::channel::<std::result::Result<Frame<Bytes>, Infallible>>(32);

        // Spawn task to convert Message stream to binary frames
        let default_msg_type = msg_type.to_string();
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

        // Build streaming request body
        let body_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let body = StreamBody::new(body_stream);

        // Build request with streaming body and trace context
        let uri = format!("http://{}{}", addr, path);
        let request = Request::builder()
            .method(Method::POST)
            .uri(&uri)
            .header(headers::MESSAGE_MODE, MessageMode::Ask.as_str())
            .header(headers::MESSAGE_TYPE, msg_type)
            .header(headers::REQUEST_TYPE, RequestType::Stream.as_str())
            .header(TRACEPARENT_HEADER, trace_ctx.to_traceparent())
            .header("content-type", "application/octet-stream")
            .body(body)
            .map_err(|e| RuntimeError::protocol_error(format!("Failed to build request: {}", e)))?;

        // Send request with timeout
        let send_future = sender.send_request(request);
        let response = tokio::time::timeout(self.config.stream_timeout, send_future)
            .await
            .map_err(|_| {
                RuntimeError::request_timeout(self.config.stream_timeout.as_millis() as u64)
            })?
            .map_err(|e| {
                RuntimeError::protocol_error(format!("Streaming request failed: {}", e))
            })?;

        Ok(response)
    }

    /// Parse HTTP response into Message (handles both single and stream responses)
    async fn parse_response(&self, response: hyper::Response<Incoming>) -> Result<Message> {
        let status = response.status();
        if !status.is_success() {
            let body = response
                .collect()
                .await
                .map_err(|e| RuntimeError::Io(e.to_string()))?
                .to_bytes();
            let error_msg = String::from_utf8_lossy(&body);
            return Err(PulsingError::from(RuntimeError::invalid_response(format!(
                "{} - {}",
                status, error_msg
            ))));
        }

        let response_type = response
            .headers()
            .get(headers::RESPONSE_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("single");

        if response_type == "stream" {
            let cancel = CancellationToken::new();
            let cancel_clone = cancel.clone();
            let stream_timeout = self.config.stream_timeout;
            let body_stream = response.into_body();
            let frame_stream =
                Self::body_to_frame_stream(body_stream, cancel_clone, stream_timeout);
            let stream_handle = StreamHandle::new(frame_stream, cancel);

            let msg_stream = stream_handle.filter_map(|result| async move {
                match result {
                    Ok(frame) => match frame.to_message() {
                        Ok(Some(msg)) => Some(Ok(msg)),
                        Ok(None) => None,
                        Err(e) => Some(Err(e)),
                    },
                    Err(e) => Some(Err(PulsingError::from(RuntimeError::Other(e.to_string())))),
                }
            });

            Ok(Message::Stream {
                default_msg_type: String::new(),
                stream: Box::pin(msg_stream),
            })
        } else {
            let body = tokio::time::timeout(self.config.request_timeout, response.collect())
                .await
                .map_err(|_| {
                    RuntimeError::request_timeout(self.config.request_timeout.as_millis() as u64)
                })?
                .map_err(|e| RuntimeError::Io(e.to_string()))?
                .to_bytes();

            Ok(Message::single("", body.to_vec()))
        }
    }

    /// Convert response body to stream of StreamFrames using binary format.
    fn body_to_frame_stream(
        body: Incoming,
        cancel: CancellationToken,
        timeout: Duration,
    ) -> impl Stream<Item = anyhow::Result<StreamFrame>> {
        let parser = Arc::new(Mutex::new(BinaryFrameParser::new()));
        let start = std::time::Instant::now();

        http_body_util::BodyStream::new(body)
            .take_while(move |_| {
                let cancelled = cancel.is_cancelled();
                let timed_out = start.elapsed() > timeout;
                async move { !cancelled && !timed_out }
            })
            .map(move |result| {
                let parser = parser.clone();
                async move {
                    let frame = result.map_err(|e| anyhow::anyhow!("Body read: {}", e))?;
                    let data = frame
                        .into_data()
                        .map_err(|_| anyhow::anyhow!("Not data frame"))?;

                    let mut parser = parser.lock().await;
                    parser.push(&data);

                    let frames = parser
                        .parse_all()
                        .into_iter()
                        .map(|r| r.map_err(|e| anyhow::anyhow!("{}", e)))
                        .collect::<Vec<_>>();
                    Ok::<_, anyhow::Error>(futures::stream::iter(frames))
                }
            })
            .buffer_unordered(1)
            .try_flatten()
    }

    /// Send a request to the given address
    async fn send_request(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
        mode: MessageMode,
    ) -> Result<hyper::Response<Incoming>> {
        let trace_ctx = TraceContext::from_current()
            .map(|p| p.child())
            .unwrap_or_default();

        let span = tracing::info_span!(
            "http.client",
            otel.name = %format!("POST {}", path),
            http.method = "POST",
            http.url = %format!("http://{}{}", addr, path),
            trace_id = %trace_ctx.trace_id,
            span_id = %trace_ctx.span_id,
        );
        let _enter = span.enter();

        let conn_guard = self.pool.get_connection(addr).await?;
        let mut conn = conn_guard.get().await;

        // Build request with trace context header
        let uri = format!("http://{}{}", addr, path);
        let request = Request::builder()
            .method(Method::POST)
            .uri(&uri)
            .header(headers::MESSAGE_MODE, mode.as_str())
            .header(headers::MESSAGE_TYPE, msg_type)
            .header(TRACEPARENT_HEADER, trace_ctx.to_traceparent())
            .header("content-type", "application/octet-stream")
            .body(Full::new(Bytes::from(payload)))
            .map_err(|e| RuntimeError::protocol_error(format!("Failed to build request: {}", e)))?;

        // Send request with timeout
        let send_future = conn.sender.send_request(request);
        let response = tokio::time::timeout(self.config.request_timeout, send_future)
            .await
            .map_err(|_| {
                RuntimeError::request_timeout(self.config.request_timeout.as_millis() as u64)
            })?
            .map_err(|e| RuntimeError::protocol_error(format!("Request failed: {}", e)))?;

        Ok(response)
    }
}

impl Clone for Http2Client {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            config: self.config.clone(),
            retry_config: self.retry_config.clone(),
            cancel: self.cancel.clone(),
            fault_injector: self.fault_injector.clone(),
        }
    }
}

/// Builder for Http2Client
pub struct Http2ClientBuilder {
    http2_config: Http2Config,
    pool_config: Option<PoolConfig>,
    retry_config: Option<RetryConfig>,
    fault_injector: Option<Arc<dyn FaultInjector>>,
}

impl Http2ClientBuilder {
    /// Create a new builder with default HTTP/2 config
    pub fn new() -> Self {
        Self {
            http2_config: Http2Config::default(),
            pool_config: None,
            retry_config: None,
            fault_injector: None,
        }
    }

    /// Set fault injector for testing / chaos engineering.
    pub fn fault_injector(mut self, injector: Option<Arc<dyn FaultInjector>>) -> Self {
        self.fault_injector = injector;
        self
    }

    /// Set HTTP/2 configuration
    pub fn http2_config(mut self, config: Http2Config) -> Self {
        self.http2_config = config;
        self
    }

    /// Set pool configuration
    pub fn pool_config(mut self, config: PoolConfig) -> Self {
        self.pool_config = Some(config);
        self
    }

    /// Set retry configuration
    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(config);
        self
    }

    /// Set maximum retries
    pub fn max_retries(mut self, n: u32) -> Self {
        self.retry_config = Some(self.retry_config.unwrap_or_default().max_retries(n));
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.http2_config.connect_timeout = timeout;
        self
    }

    /// Set request timeout
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.http2_config.request_timeout = timeout;
        self
    }

    /// Set stream timeout
    pub fn stream_timeout(mut self, timeout: Duration) -> Self {
        self.http2_config.stream_timeout = timeout;
        self
    }

    /// Build the client
    pub fn build(self) -> Http2Client {
        let client = Http2Client::with_configs(
            self.http2_config,
            self.pool_config.unwrap_or_default(),
            self.retry_config.unwrap_or_default(),
        );
        client.with_fault_injector(self.fault_injector)
    }
}

impl Default for Http2ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = Http2Client::new(Http2Config::default());
        let _ = client;
    }

    #[test]
    fn test_client_builder() {
        let client = Http2ClientBuilder::new()
            .max_retries(5)
            .connect_timeout(Duration::from_secs(10))
            .request_timeout(Duration::from_secs(60))
            .build();

        assert_eq!(client.config.connect_timeout, Duration::from_secs(10));
        assert_eq!(client.config.request_timeout, Duration::from_secs(60));
        assert_eq!(client.retry_config.max_retries, 5);
    }

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
    fn test_client_clone() {
        let client = Http2Client::new(Http2Config::default());
        let cloned = client.clone();
        // Both should share the same pool
        assert!(Arc::ptr_eq(&client.pool, &cloned.pool));
    }

    // --- 连接管理 ---

    #[test]
    fn test_client_pool_and_stats() {
        use std::sync::atomic::Ordering;
        let client = Http2Client::new(Http2Config::default());
        let pool = client.pool();
        let stats = client.stats();
        assert_eq!(stats.connections_created.load(Ordering::Relaxed), 0);
        assert_eq!(stats.connections_closed.load(Ordering::Relaxed), 0);
        assert!(Arc::ptr_eq(client.pool(), pool));
    }

    #[test]
    fn test_client_with_retry() {
        let retry = RetryConfig::no_retry();
        let client = Http2Client::new(Http2Config::default()).with_retry(retry.clone());
        assert_eq!(client.retry_config.max_retries, 0);
    }

    #[test]
    fn test_client_with_configs() {
        let http2_config = Http2Config::default();
        let pool_config = PoolConfig::default();
        let retry_config = RetryConfig::with_max_retries(2);
        let client = Http2Client::with_configs(http2_config, pool_config, retry_config);
        assert_eq!(client.retry_config.max_retries, 2);
    }

    #[tokio::test]
    async fn test_start_background_tasks_and_shutdown() {
        let client = Http2Client::new(Http2Config::default());
        client.start_background_tasks();
        client.shutdown();
        // Shutdown again should be no-op
        client.shutdown();
    }

    // --- 错误恢复：对不可达地址应返回连接错误 ---

    #[tokio::test]
    async fn test_ask_connection_error() {
        let client =
            Http2Client::new(Http2Config::default().connect_timeout(Duration::from_millis(100)))
                .with_retry(RetryConfig::no_retry());
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let result = client.ask(addr, "/actors/foo", "ping", vec![]).await;
        let err = result.unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("connection") || msg.contains("refused") || msg.contains("reset"),
            "expected connection-related error, got: {}",
            msg
        );
    }

    #[tokio::test]
    async fn test_tell_connection_error() {
        let client =
            Http2Client::new(Http2Config::default().connect_timeout(Duration::from_millis(100)))
                .with_retry(RetryConfig::no_retry());
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let result = client.tell(addr, "/actors/foo", "ping", vec![]).await;
        let err = result.unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("connection") || msg.contains("refused") || msg.contains("reset"),
            "expected connection-related error, got: {}",
            msg
        );
    }

    // --- 错误植入 ---

    #[tokio::test]
    async fn test_fault_injector_ask() {
        struct InjectAllAsk;
        impl FaultInjector for InjectAllAsk {
            fn inject(&self, ctx: &FaultInjectContext) -> Option<PulsingError> {
                if ctx.operation == FaultInjectOperation::Ask {
                    Some(PulsingError::from(RuntimeError::connection_failed(
                        ctx.addr.to_string(),
                        "injected".to_string(),
                    )))
                } else {
                    None
                }
            }
        }
        let client = Http2Client::new(Http2Config::default())
            .with_fault_injector(Some(Arc::new(InjectAllAsk)));
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let err = client.ask(addr, "/p", "t", vec![]).await.unwrap_err();
        assert!(err.to_string().to_lowercase().contains("injected"));
    }

    #[tokio::test]
    async fn test_fault_injector_tell() {
        struct InjectTell;
        impl FaultInjector for InjectTell {
            fn inject(&self, ctx: &FaultInjectContext) -> Option<PulsingError> {
                if ctx.operation == FaultInjectOperation::Tell {
                    Some(PulsingError::from(RuntimeError::request_timeout(1)))
                } else {
                    None
                }
            }
        }
        let client = Http2Client::new(Http2Config::default())
            .with_fault_injector(Some(Arc::new(InjectTell)));
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let err = client.tell(addr, "/p", "t", vec![]).await.unwrap_err();
        assert!(err.to_string().to_lowercase().contains("timeout"));
    }

    #[tokio::test]
    async fn test_fault_injector_none_no_effect() {
        let client =
            Http2Client::new(Http2Config::default().connect_timeout(Duration::from_millis(50)))
                .with_retry(RetryConfig::no_retry())
                .with_fault_injector(None);
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let result = client.ask(addr, "/p", "t", vec![]).await;
        let err = result.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("connection")
                || err.to_string().to_lowercase().contains("refused")
        );
    }
}
