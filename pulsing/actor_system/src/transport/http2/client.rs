//! HTTP/2 Client implementation
//!
//! Supports h2c (HTTP/2 over cleartext) with:
//! - Advanced connection pooling
//! - Retry strategies with exponential backoff
//! - Timeout management
//! - Streaming support

use super::config::Http2Config;
use super::pool::{ConnectionPool, PoolConfig};
use super::retry::{RetryConfig, RetryExecutor};
use super::stream::{StreamFrame, StreamHandle};
use super::{headers, MessageMode};
use crate::actor::{Message, MessageStream};
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Method, Request};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// HTTP/2 Client with connection pooling, retry, and timeout support
pub struct Http2Client {
    /// Connection pool
    pool: Arc<ConnectionPool>,
    /// HTTP/2 configuration
    config: Http2Config,
    /// Retry configuration
    retry_config: RetryConfig,
    /// Global cancellation token
    cancel: CancellationToken,
}

impl Http2Client {
    /// Create a new HTTP/2 client with default configuration
    pub fn new(config: Http2Config) -> Self {
        Self {
            pool: Arc::new(ConnectionPool::new(config.clone())),
            config,
            retry_config: RetryConfig::default(),
            cancel: CancellationToken::new(),
        }
    }

    /// Create a new HTTP/2 client with custom configurations
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
        }
    }

    /// Create client with retry configuration
    pub fn with_retry(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    /// Get the connection pool (for diagnostics)
    pub fn pool(&self) -> &Arc<ConnectionPool> {
        &self.pool
    }

    /// Get pool statistics
    pub fn stats(&self) -> &Arc<super::pool::PoolStats> {
        self.pool.stats()
    }

    /// Start background maintenance tasks
    pub fn start_background_tasks(&self) {
        self.pool.start_cleanup_task(self.cancel.clone());
    }

    /// Shutdown the client
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    /// Send an ask (request-response) message with retry
    pub async fn ask(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        let executor = RetryExecutor::new(self.retry_config.clone());

        // ask is idempotent if the message handler is idempotent
        // We treat read operations as idempotent by default
        executor
            .execute(true, || {
                self.ask_once(addr, path, msg_type, payload.clone())
            })
            .await
    }

    /// Send an ask without retry
    async fn ask_once(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Ask)
            .await?;

        let status = response.status();

        // Read response body with timeout
        let body = tokio::time::timeout(self.config.request_timeout, response.collect())
            .await
            .map_err(|_| anyhow::anyhow!("Response body read timeout"))?
            .map_err(|e| anyhow::anyhow!("Failed to read response body: {}", e))?
            .to_bytes();

        if !status.is_success() {
            let error_msg = String::from_utf8_lossy(&body);
            return Err(anyhow::anyhow!(
                "Request failed with status {}: {}",
                status,
                error_msg
            ));
        }

        Ok(body.to_vec())
    }

    /// Send a tell (fire-and-forget) message with retry
    pub async fn tell(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        let executor = RetryExecutor::new(self.retry_config.clone());

        // tell is NOT idempotent by default (could have side effects)
        executor
            .execute(false, || {
                self.tell_once(addr, path, msg_type, payload.clone())
            })
            .await
    }

    /// Send a tell without retry
    async fn tell_once(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Tell)
            .await?;

        let status = response.status();

        if !status.is_success() {
            let body = response.collect().await?.to_bytes();
            let error_msg = String::from_utf8_lossy(&body);
            return Err(anyhow::anyhow!(
                "Tell failed with status {}: {}",
                status,
                error_msg
            ));
        }

        Ok(())
    }

    /// Send a stream request and receive streaming response as StreamFrame
    ///
    /// Note: Streaming requests are NOT retried (they are not idempotent)
    pub async fn ask_stream(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<StreamHandle<StreamFrame>> {
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Stream)
            .await?;

        let status = response.status();

        if !status.is_success() {
            let body = response.collect().await?.to_bytes();
            let error_msg = String::from_utf8_lossy(&body);
            return Err(anyhow::anyhow!(
                "Stream request failed with status {}: {}",
                status,
                error_msg
            ));
        }

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        // Apply stream timeout
        let stream_timeout = self.config.stream_timeout;
        let body_stream = response.into_body();
        let frame_stream = Self::body_to_frame_stream(body_stream, cancel_clone, stream_timeout);

        Ok(StreamHandle::new(frame_stream, cancel))
    }

    /// Send a stream request and receive streaming response as MessageStream
    pub async fn ask_stream_raw(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<MessageStream> {
        let stream_handle = self.ask_stream(addr, path, msg_type, payload).await?;

        // Convert StreamFrame stream to Message stream
        let msg_stream = stream_handle.filter_map(|result| async move {
            match result {
                Ok(frame) => {
                    // Skip end frames with no data
                    if frame.end && frame.data.is_empty() {
                        return None;
                    }

                    // Check for errors
                    if let Some(error) = frame.error {
                        return Some(Err(anyhow::anyhow!("{}", error)));
                    }

                    // Decode data
                    match frame.decode_data() {
                        Ok(payload) => Some(Ok(Message::single(&frame.msg_type, payload))),
                        Err(e) => Some(Err(e)),
                    }
                }
                Err(e) => Some(Err(e)),
            }
        });

        Ok(Box::pin(msg_stream))
    }

    /// Convert response body to stream of StreamFrames with timeout
    fn body_to_frame_stream(
        body: Incoming,
        cancel: CancellationToken,
        timeout: Duration,
    ) -> impl Stream<Item = anyhow::Result<StreamFrame>> {
        let buffer = Arc::new(Mutex::new(String::new()));
        let start = std::time::Instant::now();

        http_body_util::BodyStream::new(body)
            .take_while(move |_| {
                let cancelled = cancel.is_cancelled();
                let timed_out = start.elapsed() > timeout;
                async move { !cancelled && !timed_out }
            })
            .map(move |result| {
                let buffer = buffer.clone();
                async move {
                    let frame = result.map_err(|e| anyhow::anyhow!("Body read error: {}", e))?;
                    let data = frame
                        .into_data()
                        .map_err(|_| anyhow::anyhow!("Not data frame"))?;

                    let mut buf = buffer.lock().await;
                    buf.push_str(&String::from_utf8_lossy(&data));

                    // Extract complete lines
                    let mut frames = Vec::new();
                    while let Some(newline_pos) = buf.find('\n') {
                        let line = buf.drain(..=newline_pos).collect::<String>();
                        let line = line.trim();
                        if !line.is_empty() {
                            match StreamFrame::from_ndjson(line) {
                                Ok(frame) => frames.push(Ok(frame)),
                                Err(e) => frames.push(Err(anyhow::anyhow!("Parse error: {}", e))),
                            }
                        }
                    }

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
    ) -> anyhow::Result<hyper::Response<Incoming>> {
        let conn_guard = self.pool.get_connection(addr).await?;
        let mut conn = conn_guard.get().await;

        // Build request
        let uri = format!("http://{}{}", addr, path);
        let request = Request::builder()
            .method(Method::POST)
            .uri(&uri)
            .header(headers::MESSAGE_MODE, mode.as_str())
            .header(headers::MESSAGE_TYPE, msg_type)
            .header("content-type", "application/octet-stream")
            .body(Full::new(Bytes::from(payload)))
            .map_err(|e| anyhow::anyhow!("Failed to build request: {}", e))?;

        // Send request with timeout
        let send_future = conn.sender.send_request(request);
        let response = tokio::time::timeout(self.config.request_timeout, send_future)
            .await
            .map_err(|_| anyhow::anyhow!("Request timeout"))?
            .map_err(|e| anyhow::anyhow!("Request failed: {}", e))?;

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
        }
    }
}

/// Builder for Http2Client
pub struct Http2ClientBuilder {
    http2_config: Http2Config,
    pool_config: Option<PoolConfig>,
    retry_config: Option<RetryConfig>,
}

impl Http2ClientBuilder {
    /// Create a new builder with default HTTP/2 config
    pub fn new() -> Self {
        Self {
            http2_config: Http2Config::default(),
            pool_config: None,
            retry_config: None,
        }
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
        Http2Client::with_configs(
            self.http2_config,
            self.pool_config.unwrap_or_default(),
            self.retry_config.unwrap_or_default(),
        )
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

        assert_eq!(MessageMode::from_str("ask"), Some(MessageMode::Ask));
        assert_eq!(MessageMode::from_str("TELL"), Some(MessageMode::Tell));
        assert_eq!(MessageMode::from_str("Stream"), Some(MessageMode::Stream));
        assert_eq!(MessageMode::from_str("invalid"), None);
    }

    #[test]
    fn test_client_clone() {
        let client = Http2Client::new(Http2Config::default());
        let cloned = client.clone();
        // Both should share the same pool
        assert!(Arc::ptr_eq(&client.pool, &cloned.pool));
    }
}
