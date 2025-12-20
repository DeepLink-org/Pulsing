//! HTTP/2 Client implementation
//!
//! Supports h2c (HTTP/2 over cleartext) with connection pooling.

use super::config::Http2Config;
use super::stream::{StreamFrame, StreamHandle};
use super::{headers, MessageMode};
use crate::actor::{Message, MessageStream};
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http2;
use hyper::{Method, Request};
use hyper_util::rt::{TokioExecutor, TokioIo};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

/// HTTP/2 connection pool entry
struct PooledConnection {
    sender: http2::SendRequest<Full<Bytes>>,
    #[allow(dead_code)]
    created_at: std::time::Instant,
}

/// Connection pool
struct ConnectionPool {
    connections: RwLock<HashMap<SocketAddr, Vec<Arc<Mutex<PooledConnection>>>>>,
    config: Http2Config,
}

impl ConnectionPool {
    fn new(config: Http2Config) -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Get or create a connection to the given address
    async fn get_connection(
        &self,
        addr: SocketAddr,
    ) -> anyhow::Result<Arc<Mutex<PooledConnection>>> {
        // Try to get an existing connection
        {
            let connections = self.connections.read().await;
            if let Some(pool) = connections.get(&addr) {
                for conn in pool.iter() {
                    if let Ok(guard) = conn.try_lock() {
                        if guard.sender.is_ready() {
                            drop(guard);
                            return Ok(conn.clone());
                        }
                    }
                }
            }
        }

        // Create a new connection
        let conn = self.create_connection(addr).await?;
        let conn = Arc::new(Mutex::new(conn));

        // Add to pool
        {
            let mut connections = self.connections.write().await;
            let pool = connections.entry(addr).or_insert_with(Vec::new);

            // Limit pool size
            if pool.len() < self.config.max_connections_per_host {
                pool.push(conn.clone());
            }
        }

        Ok(conn)
    }

    /// Create a new HTTP/2 connection
    async fn create_connection(&self, addr: SocketAddr) -> anyhow::Result<PooledConnection> {
        let stream = tokio::time::timeout(self.config.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| anyhow::anyhow!("Connection timeout"))?
            .map_err(|e| anyhow::anyhow!("Connection failed: {}", e))?;

        let io = TokioIo::new(stream);

        // Create HTTP/2 connection with prior knowledge (h2c)
        let (sender, conn) = http2::handshake(TokioExecutor::new(), io)
            .await
            .map_err(|e| anyhow::anyhow!("HTTP/2 handshake failed: {}", e))?;

        // Spawn connection driver
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                tracing::debug!(error = %e, "HTTP/2 connection closed");
            }
        });

        Ok(PooledConnection {
            sender,
            created_at: std::time::Instant::now(),
        })
    }
}

/// HTTP/2 Client
pub struct Http2Client {
    pool: Arc<ConnectionPool>,
    config: Http2Config,
}

impl Http2Client {
    /// Create a new HTTP/2 client
    pub fn new(config: Http2Config) -> Self {
        Self {
            pool: Arc::new(ConnectionPool::new(config.clone())),
            config,
        }
    }

    /// Send an ask (request-response) message
    pub async fn ask(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Ask)
            .await?;

        // Get status before consuming body
        let status = response.status();

        // Read response body
        let body = response.collect().await?.to_bytes();

        // Check status
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

    /// Send a tell (fire-and-forget) message
    pub async fn tell(
        &self,
        addr: SocketAddr,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        let response = self
            .send_request(addr, path, msg_type, payload, MessageMode::Tell)
            .await?;

        // Get status before consuming body
        let status = response.status();

        // Check status
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

        // Get status before consuming body
        let status = response.status();

        // Check status
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

        // Convert response body to stream of frames
        let body_stream = response.into_body();
        let frame_stream = Self::body_to_frame_stream(body_stream, cancel_clone);

        Ok(StreamHandle::new(frame_stream, cancel))
    }

    /// Send a stream request and receive streaming response as MessageStream
    ///
    /// This is a convenience method that wraps `ask_stream` and converts
    /// StreamFrames to Messages.
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

    /// Convert response body to stream of StreamFrames
    fn body_to_frame_stream(
        body: Incoming,
        cancel: CancellationToken,
    ) -> impl Stream<Item = anyhow::Result<StreamFrame>> {
        // Buffer for partial lines
        let buffer = Arc::new(Mutex::new(String::new()));

        http_body_util::BodyStream::new(body)
            .take_while(move |_| {
                let cancelled = cancel.is_cancelled();
                async move { !cancelled }
            })
            .map(move |result| {
                let buffer = buffer.clone();
                async move {
                    let frame = result.map_err(|e| anyhow::anyhow!("Body read error: {}", e))?;
                    let data = frame.into_data().map_err(|_| anyhow::anyhow!("Not data frame"))?;

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
        let conn = self.pool.get_connection(addr).await?;
        let mut guard = conn.lock().await;

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
        let send_future = guard.sender.send_request(request);
        let response: hyper::Response<Incoming> =
            tokio::time::timeout(self.config.request_timeout, send_future)
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = Http2Client::new(Http2Config::default());
        // Just test that it compiles and can be created
        let _ = client;
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
}
