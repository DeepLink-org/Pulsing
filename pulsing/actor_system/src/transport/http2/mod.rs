//! HTTP/2 Transport Layer
//!
//! Provides HTTP/2 (h2c - cleartext) transport for actor communication with streaming support.
//!
//! ## Features
//!
//! - HTTP/2 over cleartext (h2c) - no TLS required
//! - Streaming responses via `ask_stream`
//! - Connection multiplexing
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

mod client;
mod config;
mod server;
mod stream;

pub use client::Http2Client;
pub use config::Http2Config;
pub use server::{Http2Server, Http2ServerHandler};
pub use stream::{StreamFrame, StreamHandle};

use crate::actor::{ActorId, ActorPath, Message, PayloadStream, RemoteTransport};
use futures::StreamExt;
use std::net::SocketAddr;
use std::sync::Arc;

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

    pub fn from_str(s: &str) -> Option<Self> {
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
    pub const REQUEST_ID: &str = "x-request-id";
}

/// Target type for HTTP/2 remote transport
#[derive(Clone)]
enum Http2RemoteTarget {
    /// Target by actor name
    Actor(String),
    /// Target by named actor path
    Named(ActorPath),
}

/// HTTP/2 Remote Transport for ActorRef
///
/// Implements the `RemoteTransport` trait, enabling `ActorRef` to communicate
/// with remote actors over HTTP/2, including streaming support.
pub struct Http2RemoteTransport {
    client: Arc<Http2Client>,
    remote_addr: SocketAddr,
    target: Http2RemoteTarget,
}

impl Http2RemoteTransport {
    /// Create a new remote transport targeting an actor by name
    pub fn new(client: Arc<Http2Client>, remote_addr: SocketAddr, actor_name: String) -> Self {
        Self {
            client,
            remote_addr,
            target: Http2RemoteTarget::Actor(actor_name),
        }
    }

    /// Create a new remote transport targeting a named actor by path
    pub fn new_named(client: Arc<Http2Client>, remote_addr: SocketAddr, path: ActorPath) -> Self {
        Self {
            client,
            remote_addr,
            target: Http2RemoteTarget::Named(path),
        }
    }

    /// Get the path for the request
    fn request_path(&self) -> String {
        match &self.target {
            Http2RemoteTarget::Actor(name) => format!("/actors/{}", name),
            Http2RemoteTarget::Named(path) => format!("/named/{}", path.as_str()),
        }
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
        let path = self.request_path();
        self.client.ask(self.remote_addr, &path, msg_type, payload).await
    }

    async fn send(
        &self,
        _actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        let path = self.request_path();
        self.client.tell(self.remote_addr, &path, msg_type, payload).await
    }

    async fn request_stream(
        &self,
        _actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<PayloadStream> {
        let path = self.request_path();
        let msg_stream = self
            .client
            .ask_stream_raw(self.remote_addr, &path, msg_type, payload)
            .await?;
        // Convert MessageStream to PayloadStream by extracting payload
        let payload_stream = msg_stream.map(|result| {
            result.and_then(|msg| {
                let Message::Single { data, .. } = msg else {
                    return Err(anyhow::anyhow!("Expected single message in stream"));
                };
                Ok(data)
            })
        });
        Ok(Box::pin(payload_stream))
    }
}
