//! Out-cluster connector for accessing actors without joining the cluster.
//!
//! `PulsingConnect` connects to a cluster gateway node via HTTP/2 and
//! transparently routes requests to any actor in the cluster. It does not
//! participate in gossip, does not register as a node, and does not run
//! fault detection — keeping the connector extremely lightweight.
//!
//! # Example
//!
//! ```rust,ignore
//! let conn = PulsingConnect::connect("10.0.1.1:8080".parse()?).await?;
//! let counter = conn.resolve("services/counter").await?;
//! let result = counter.ask("increment", vec![]).await?;
//! conn.close();
//! ```

mod reference;

pub use reference::ConnectActorRef;

use crate::error::{PulsingError, Result, RuntimeError};
use crate::transport::{Http2Client, Http2Config};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Resolve response from the gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveResponse {
    pub found: bool,
    pub path: String,
    pub gateway: String,
    pub instance_count: usize,
}

/// Gateway member list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembersResponse {
    pub gateways: Vec<String>,
}

/// Out-cluster connector — connects to a gateway node without joining the cluster.
pub struct PulsingConnect {
    active_gateway: RwLock<SocketAddr>,
    gateways: RwLock<Vec<SocketAddr>>,
    http_client: Arc<Http2Client>,
}

impl PulsingConnect {
    /// Connect to a single gateway address.
    pub async fn connect(gateway_addr: SocketAddr) -> Result<Arc<Self>> {
        Self::connect_multi(vec![gateway_addr]).await
    }

    /// Connect with multiple gateway addresses for failover.
    pub async fn connect_multi(gateway_addrs: Vec<SocketAddr>) -> Result<Arc<Self>> {
        if gateway_addrs.is_empty() {
            return Err(PulsingError::from(RuntimeError::Other(
                "At least one gateway address is required".into(),
            )));
        }

        let http_client = Arc::new(Http2Client::new(Http2Config::default()));
        http_client.start_background_tasks();

        let active = gateway_addrs[0];
        let conn = Arc::new(Self {
            active_gateway: RwLock::new(active),
            gateways: RwLock::new(gateway_addrs),
            http_client,
        });

        // Try to refresh gateway list from the cluster
        let _ = conn.refresh_gateways().await;

        Ok(conn)
    }

    /// Resolve a named actor via the gateway.
    pub async fn resolve(&self, path: &str) -> Result<ConnectActorRef> {
        let gateway = *self.active_gateway.read().await;
        let body = serde_json::to_vec(&serde_json::json!({"path": path}))
            .map_err(|e| PulsingError::from(RuntimeError::Other(e.to_string())))?;

        let resp_bytes = self.ask_gateway(gateway, "/client/resolve", &body).await?;

        let resp: ResolveResponse = serde_json::from_slice(&resp_bytes)
            .map_err(|e| PulsingError::from(RuntimeError::Other(e.to_string())))?;

        if !resp.found {
            return Err(PulsingError::from(RuntimeError::named_actor_not_found(
                path,
            )));
        }

        Ok(ConnectActorRef::new(
            gateway,
            path.to_string(),
            self.http_client.clone(),
        ))
    }

    /// Refresh the gateway list from the current active gateway.
    pub async fn refresh_gateways(&self) -> Result<()> {
        let gateway = *self.active_gateway.read().await;
        let resp_bytes = self.ask_gateway(gateway, "/client/members", &[]).await?;

        let resp: MembersResponse = serde_json::from_slice(&resp_bytes)
            .map_err(|e| PulsingError::from(RuntimeError::Other(e.to_string())))?;

        let mut new_gateways = Vec::new();
        for addr_str in &resp.gateways {
            if let Ok(addr) = addr_str.parse::<SocketAddr>() {
                new_gateways.push(addr);
            }
        }

        if !new_gateways.is_empty() {
            *self.gateways.write().await = new_gateways;
        }

        Ok(())
    }

    /// Switch to the next available gateway (for failover).
    pub async fn failover(&self) -> Result<()> {
        let current = *self.active_gateway.read().await;
        let gateways = self.gateways.read().await;

        let next = gateways
            .iter()
            .find(|&&addr| addr != current)
            .ok_or_else(|| {
                PulsingError::from(RuntimeError::Other(
                    "No alternative gateway available".into(),
                ))
            })?;

        *self.active_gateway.write().await = *next;
        Ok(())
    }

    /// Get the current active gateway address.
    pub async fn active_gateway(&self) -> SocketAddr {
        *self.active_gateway.read().await
    }

    /// Get all known gateway addresses.
    pub async fn gateway_list(&self) -> Vec<SocketAddr> {
        self.gateways.read().await.clone()
    }

    /// Shutdown the connector.
    pub fn close(&self) {
        self.http_client.shutdown();
    }

    /// Internal: send a GET/POST to a gateway management endpoint.
    async fn ask_gateway(&self, gateway: SocketAddr, path: &str, body: &[u8]) -> Result<Vec<u8>> {
        self.http_client
            .ask(gateway, path, "client", body.to_vec())
            .await
    }

    /// Get a reference to the underlying HTTP/2 client.
    pub fn http_client(&self) -> &Arc<Http2Client> {
        &self.http_client
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_response_deserialize() {
        let json = r#"{"found":true,"path":"services/counter","gateway":"10.0.1.1:8080","instance_count":2}"#;
        let resp: ResolveResponse = serde_json::from_str(json).unwrap();
        assert!(resp.found);
        assert_eq!(resp.path, "services/counter");
        assert_eq!(resp.instance_count, 2);
    }

    #[test]
    fn test_members_response_deserialize() {
        let json = r#"{"gateways":["10.0.1.1:8080","10.0.1.2:8080"]}"#;
        let resp: MembersResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.gateways.len(), 2);
    }

    #[tokio::test]
    async fn test_connect_empty_addrs() {
        let result = PulsingConnect::connect_multi(vec![]).await;
        assert!(result.is_err());
    }
}
