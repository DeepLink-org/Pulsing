//! TCP transport for actor communication

use super::codec::{MessageCodec, TransportMessage};
use crate::actor::{ActorId, PayloadStream, RemoteTransport};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_util::codec::Framed;
use tokio_util::sync::CancellationToken;

/// TCP message handler trait (local to this module)
#[async_trait::async_trait]
pub trait TcpMessageHandler: Send + Sync + 'static {
    /// Handle an incoming message for an actor
    async fn handle_message(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>>;
}

/// TCP transport configuration
#[derive(Clone, Debug)]
pub struct TcpTransportConfig {
    /// Request timeout
    pub request_timeout: Duration,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Max connections per peer
    pub max_connections_per_peer: usize,

    /// Keepalive interval
    pub keepalive_interval: Duration,
}

impl Default for TcpTransportConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
            max_connections_per_peer: 4,
            keepalive_interval: Duration::from_secs(30),
        }
    }
}

/// TCP transport layer
pub struct TcpTransport {
    /// Local bind address
    local_addr: SocketAddr,

    /// Configuration
    config: TcpTransportConfig,

    /// Pending requests (id -> response sender)
    pending: Arc<DashMap<u64, oneshot::Sender<Result<Vec<u8>, String>>>>,

    /// Message handler for incoming requests
    handler: Arc<dyn TcpMessageHandler>,

    /// Connection pool (addr -> connection)
    connections: Arc<DashMap<SocketAddr, Arc<Connection>>>,

    /// Cancellation token
    cancel: CancellationToken,
}

/// A single TCP connection
struct Connection {
    /// Sender for outgoing messages
    sender: tokio::sync::mpsc::Sender<TransportMessage>,
}

impl TcpTransport {
    /// Create a new TCP transport
    pub async fn new(
        bind_addr: SocketAddr,
        handler: Arc<dyn TcpMessageHandler>,
        config: TcpTransportConfig,
    ) -> anyhow::Result<Arc<Self>> {
        let listener = TcpListener::bind(bind_addr).await?;
        let local_addr = listener.local_addr()?;

        let cancel = CancellationToken::new();

        let transport = Arc::new(Self {
            local_addr,
            config,
            pending: Arc::new(DashMap::new()),
            handler,
            connections: Arc::new(DashMap::new()),
            cancel: cancel.clone(),
        });

        // Start listener task
        let transport_clone = transport.clone();
        tokio::spawn(async move {
            transport_clone.accept_loop(listener).await;
        });

        tracing::info!(addr = %local_addr, "TCP transport started");

        Ok(transport)
    }

    /// Get local address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Accept incoming connections
    async fn accept_loop(self: Arc<Self>, listener: TcpListener) {
        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, peer_addr)) => {
                            tracing::debug!(peer = %peer_addr, "Accepted connection");
                            let transport = self.clone();
                            tokio::spawn(async move {
                                transport.handle_connection(stream, peer_addr).await;
                            });
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Accept error");
                        }
                    }
                }
                _ = self.cancel.cancelled() => {
                    tracing::info!("TCP accept loop shutting down");
                    break;
                }
            }
        }
    }

    /// Handle an incoming connection
    async fn handle_connection(&self, stream: TcpStream, peer_addr: SocketAddr) {
        let mut framed = Framed::new(stream, MessageCodec::new());

        loop {
            tokio::select! {
                result = framed.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            self.handle_message(msg, &mut framed).await;
                        }
                        Some(Err(e)) => {
                            tracing::warn!(peer = %peer_addr, error = %e, "Read error");
                            break;
                        }
                        None => {
                            tracing::debug!(peer = %peer_addr, "Connection closed");
                            break;
                        }
                    }
                }
                _ = self.cancel.cancelled() => {
                    break;
                }
            }
        }
    }

    /// Handle an incoming message
    async fn handle_message(
        &self,
        msg: TransportMessage,
        framed: &mut Framed<TcpStream, MessageCodec>,
    ) {
        match msg {
            TransportMessage::Request {
                id,
                actor_id,
                msg_type,
                payload,
            } => {
                let result = self
                    .handler
                    .handle_message(&actor_id, &msg_type, payload)
                    .await;

                let response = TransportMessage::response(id, result.map_err(|e| e.to_string()));

                if let Err(e) = framed.send(response).await {
                    tracing::warn!(error = %e, "Failed to send response");
                }
            }

            TransportMessage::Response { id, result } => {
                if let Some((_, tx)) = self.pending.remove(&id) {
                    let _ = tx.send(result);
                }
            }

            TransportMessage::OneWay {
                actor_id,
                msg_type,
                payload,
            } => {
                let _ = self
                    .handler
                    .handle_message(&actor_id, &msg_type, payload)
                    .await;
            }

            TransportMessage::Ping { seq } => {
                let _ = framed.send(TransportMessage::Pong { seq }).await;
            }

            TransportMessage::Pong { seq: _ } => {
                // Handled by keepalive
            }
        }
    }

    /// Get or create a connection to a peer
    async fn get_connection(&self, addr: SocketAddr) -> anyhow::Result<Arc<Connection>> {
        // Check existing connection
        if let Some(conn) = self.connections.get(&addr) {
            return Ok(conn.clone());
        }

        // Create new connection
        let stream = tokio::time::timeout(self.config.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| anyhow::anyhow!("Connection timeout"))??;

        let mut framed = Framed::new(stream, MessageCodec::new());

        // Create channel for sending
        let (tx, mut rx) = tokio::sync::mpsc::channel::<TransportMessage>(64);

        let conn = Arc::new(Connection { sender: tx });
        self.connections.insert(addr, conn.clone());

        // Spawn send/receive tasks
        let pending = self.pending.clone();
        let handler = self.handler.clone();
        let cancel = self.cancel.clone();
        let connections = self.connections.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Send outgoing messages
                    Some(msg) = rx.recv() => {
                        if let Err(e) = framed.send(msg).await {
                            tracing::warn!(error = %e, peer = %addr, "Send error");
                            break;
                        }
                    }

                    // Receive incoming messages
                    result = framed.next() => {
                        match result {
                            Some(Ok(msg)) => {
                                match msg {
                                    TransportMessage::Response { id, result } => {
                                        if let Some((_, tx)) = pending.remove(&id) {
                                            let _ = tx.send(result);
                                        }
                                    }
                                    TransportMessage::Request { id, actor_id, msg_type, payload } => {
                                        let result = handler.handle_message(&actor_id, &msg_type, payload).await;
                                        let response = TransportMessage::response(id, result.map_err(|e| e.to_string()));
                                        if let Err(e) = framed.send(response).await {
                                            tracing::warn!(error = %e, "Failed to send response");
                                        }
                                    }
                                    TransportMessage::OneWay { actor_id, msg_type, payload } => {
                                        let _ = handler.handle_message(&actor_id, &msg_type, payload).await;
                                    }
                                    TransportMessage::Ping { seq } => {
                                        let _ = framed.send(TransportMessage::Pong { seq }).await;
                                    }
                                    TransportMessage::Pong { .. } => {}
                                }
                            }
                            Some(Err(e)) => {
                                tracing::warn!(error = %e, peer = %addr, "Read error");
                                break;
                            }
                            None => {
                                tracing::debug!(peer = %addr, "Connection closed");
                                break;
                            }
                        }
                    }

                    _ = cancel.cancelled() => {
                        break;
                    }
                }
            }

            // Cleanup
            connections.remove(&addr);
        });

        Ok(conn)
    }

    /// Send a request and wait for response
    async fn send_request(
        &self,
        addr: SocketAddr,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        let conn = self.get_connection(addr).await?;

        let (id, msg) = TransportMessage::request(actor_id.clone(), msg_type.to_string(), payload);

        let (tx, rx) = oneshot::channel();
        self.pending.insert(id, tx);

        // Send request
        conn.sender
            .send(msg)
            .await
            .map_err(|_| anyhow::anyhow!("Connection closed"))?;

        // Wait for response
        let result = tokio::time::timeout(self.config.request_timeout, rx)
            .await
            .map_err(|_| anyhow::anyhow!("Request timeout"))?
            .map_err(|_| anyhow::anyhow!("Request cancelled"))?;

        result.map_err(|e| anyhow::anyhow!("Remote error: {}", e))
    }

    /// Send a one-way message
    async fn send_one_way(
        &self,
        addr: SocketAddr,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        let conn = self.get_connection(addr).await?;

        let msg = TransportMessage::one_way(actor_id.clone(), msg_type.to_string(), payload);

        conn.sender
            .send(msg)
            .await
            .map_err(|_| anyhow::anyhow!("Connection closed"))?;

        Ok(())
    }

    /// Shutdown the transport
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }
}

/// Remote transport implementation for TCP
pub struct TcpRemoteTransport {
    transport: Arc<TcpTransport>,
    remote_addr: SocketAddr,
}

impl TcpRemoteTransport {
    pub fn new(transport: Arc<TcpTransport>, remote_addr: SocketAddr) -> Self {
        Self {
            transport,
            remote_addr,
        }
    }
}

#[async_trait::async_trait]
impl RemoteTransport for TcpRemoteTransport {
    async fn request(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        self.transport
            .send_request(self.remote_addr, actor_id, msg_type, payload)
            .await
    }

    async fn send(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        self.transport
            .send_one_way(self.remote_addr, actor_id, msg_type, payload)
            .await
    }

    async fn request_stream(
        &self,
        _actor_id: &ActorId,
        _msg_type: &str,
        _payload: Vec<u8>,
    ) -> anyhow::Result<PayloadStream> {
        // TCP transport does not support streaming.
        // Use HTTP/2 transport for streaming support.
        Err(anyhow::anyhow!(
            "Streaming not supported with TCP transport. Use HTTP/2 transport instead."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoHandler;

    #[async_trait::async_trait]
    impl TcpMessageHandler for EchoHandler {
        async fn handle_message(
            &self,
            _actor_id: &ActorId,
            _msg_type: &str,
            payload: Vec<u8>,
        ) -> anyhow::Result<Vec<u8>> {
            // Echo back the payload
            Ok(payload)
        }
    }

    #[tokio::test]
    async fn test_tcp_transport_echo() {
        let handler = Arc::new(EchoHandler);

        // Start server
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let server = TcpTransport::new(server_addr, handler.clone(), TcpTransportConfig::default())
            .await
            .unwrap();

        let server_addr = server.local_addr();

        // Start client
        let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let client = TcpTransport::new(client_addr, handler, TcpTransportConfig::default())
            .await
            .unwrap();

        // Send request
        let actor_id = ActorId::local("test");
        let response = client
            .send_request(server_addr, &actor_id, "test", vec![1, 2, 3])
            .await
            .unwrap();

        assert_eq!(response, vec![1, 2, 3]);

        // Cleanup
        server.shutdown();
        client.shutdown();
    }
}
