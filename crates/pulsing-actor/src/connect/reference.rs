//! Actor reference for out-cluster connectors.
//!
//! `ConnectActorRef` routes all requests through a gateway node using the
//! `/named/{path}` endpoint. The gateway handles internal cluster routing
//! transparently.

use crate::actor::Message;
use crate::error::Result;
use crate::transport::{Http2Client, StreamFrame, StreamHandle};
use std::net::SocketAddr;
use std::sync::Arc;

/// Actor reference used by out-cluster connectors.
///
/// All requests are routed through the gateway node. The connector never
/// communicates directly with the target actor's node.
#[derive(Clone)]
pub struct ConnectActorRef {
    gateway: SocketAddr,
    path: String,
    http_client: Arc<Http2Client>,
}

impl ConnectActorRef {
    pub fn new(gateway: SocketAddr, path: String, http_client: Arc<Http2Client>) -> Self {
        Self {
            gateway,
            path,
            http_client,
        }
    }

    /// Send a request and wait for a response.
    pub async fn ask(&self, msg_type: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        let url_path = format!("/named/{}", self.path);
        self.http_client
            .ask(self.gateway, &url_path, msg_type, payload)
            .await
    }

    /// Fire-and-forget message.
    pub async fn tell(&self, msg_type: &str, payload: Vec<u8>) -> Result<()> {
        let url_path = format!("/named/{}", self.path);
        self.http_client
            .tell(self.gateway, &url_path, msg_type, payload)
            .await
    }

    /// Streaming request — returns a stream of frames.
    pub async fn ask_stream(
        &self,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<StreamHandle<StreamFrame>> {
        let url_path = format!("/named/{}", self.path);
        self.http_client
            .ask_stream(self.gateway, &url_path, msg_type, payload)
            .await
    }

    /// Send a full `Message` (single or stream) and receive a `Message` response.
    pub async fn send_message(&self, msg: Message) -> Result<Message> {
        let url_path = format!("/named/{}", self.path);
        self.http_client
            .send_message_full(self.gateway, &url_path, msg)
            .await
    }

    /// Get the gateway address this reference routes through.
    pub fn gateway(&self) -> SocketAddr {
        self.gateway
    }

    /// Get the actor path.
    pub fn path(&self) -> &str {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::Http2Config;

    #[test]
    fn test_connect_actor_ref_accessors() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let r = ConnectActorRef::new(addr, "services/counter".into(), client);
        assert_eq!(r.gateway(), addr);
        assert_eq!(r.path(), "services/counter");
    }

    #[test]
    fn test_connect_actor_ref_clone() {
        let client = Arc::new(Http2Client::new(Http2Config::default()));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let r = ConnectActorRef::new(addr, "services/counter".into(), client);
        let r2 = r.clone();
        assert_eq!(r.gateway(), r2.gateway());
        assert_eq!(r.path(), r2.path());
    }
}
