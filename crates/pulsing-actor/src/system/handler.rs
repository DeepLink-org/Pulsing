//! HTTP/2 message handler for the actor system

use super::handle::LocalActorHandle;
use crate::actor::{ActorId, ActorPath, Envelope, Message, NodeId};
use crate::cluster::backends::{RegisterActorRequest, UnregisterActorRequest};
use crate::cluster::{GossipBackend, GossipMessage, HeadNodeBackend, NamingBackend};
use crate::error::{PulsingError, RuntimeError};
use crate::metrics::{metrics, SystemMetrics as PrometheusMetrics};
use crate::transport::Http2ServerHandler;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Unified message handler for HTTP/2 transport
pub(crate) struct SystemMessageHandler {
    node_id: NodeId,
    /// Local actors indexed by ActorId
    local_actors: Arc<DashMap<ActorId, LocalActorHandle>>,
    /// Actor name to ActorId mapping
    actor_names: Arc<DashMap<String, ActorId>>,
    named_actor_paths: Arc<DashMap<String, String>>,
    cluster: Arc<RwLock<Option<Arc<dyn NamingBackend>>>>,
}

impl SystemMessageHandler {
    pub fn new(
        node_id: NodeId,
        local_actors: Arc<DashMap<ActorId, LocalActorHandle>>,
        actor_names: Arc<DashMap<String, ActorId>>,
        named_actor_paths: Arc<DashMap<String, String>>,
        cluster: Arc<RwLock<Option<Arc<dyn NamingBackend>>>>,
    ) -> Self {
        Self {
            node_id,
            local_actors,
            actor_names,
            named_actor_paths,
            cluster,
        }
    }

    /// Find actor sender by name or ActorId (O(1) lookup)
    fn find_actor_sender(&self, actor_name: &str) -> anyhow::Result<mpsc::Sender<Envelope>> {
        // First try by name -> ActorId -> handle
        if let Some(actor_id) = self.actor_names.get(actor_name) {
            if let Some(handle) = self.local_actors.get(actor_id.value()) {
                return Ok(handle.sender.clone());
            }
        }

        // Then try parsing as ActorId (UUID format)
        if let Ok(uuid) = uuid::Uuid::parse_str(actor_name) {
            let actor_id = ActorId::new(uuid.as_u128());
            if let Some(handle) = self.local_actors.get(&actor_id) {
                return Ok(handle.sender.clone());
            }
        }

        Err(anyhow::Error::from(PulsingError::from(
            RuntimeError::actor_not_found(actor_name.to_string()),
        )))
    }

    /// Dispatch a message to an actor (ask pattern)
    async fn dispatch_message(&self, path: &str, msg: Message) -> anyhow::Result<Message> {
        if let Some(actor_name) = path.strip_prefix("/actors/") {
            self.send_to_local_actor(actor_name, msg).await
        } else if let Some(named_path) = path.strip_prefix("/named/") {
            self.send_to_named_actor(named_path, msg).await
        } else {
            Err(anyhow::anyhow!("Invalid path: {}", path))
        }
    }

    /// Dispatch a fire-and-forget message
    async fn dispatch_tell(&self, path: &str, msg: Message) -> anyhow::Result<()> {
        if let Some(actor_name) = path.strip_prefix("/actors/") {
            self.tell_local_actor(actor_name, msg).await
        } else if let Some(named_path) = path.strip_prefix("/named/") {
            self.tell_named_actor(named_path, msg).await
        } else {
            Err(anyhow::anyhow!("Invalid path: {}", path))
        }
    }

    async fn send_to_local_actor(&self, actor_name: &str, msg: Message) -> anyhow::Result<Message> {
        let sender = self.find_actor_sender(actor_name)?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let envelope = Envelope::ask(msg, tx);

        sender
            .send(envelope)
            .await
            .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

        rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?
    }

    async fn tell_local_actor(&self, actor_name: &str, msg: Message) -> anyhow::Result<()> {
        let sender = self.find_actor_sender(actor_name)?;
        let envelope = Envelope::tell(msg);

        sender
            .send(envelope)
            .await
            .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

        Ok(())
    }

    async fn send_to_named_actor(&self, path: &str, msg: Message) -> anyhow::Result<Message> {
        let actor_name = self
            .named_actor_paths
            .get(path)
            .ok_or_else(|| anyhow::anyhow!("Named actor not found: {}", path))?
            .clone();

        self.send_to_local_actor(&actor_name, msg).await
    }

    async fn tell_named_actor(&self, path: &str, msg: Message) -> anyhow::Result<()> {
        let actor_name = self
            .named_actor_paths
            .get(path)
            .ok_or_else(|| anyhow::anyhow!("Named actor not found: {}", path))?
            .clone();

        self.tell_local_actor(&actor_name, msg).await
    }
}

#[async_trait::async_trait]
impl Http2ServerHandler for SystemMessageHandler {
    /// Unified message handler - accepts Message (Single or Stream), returns Message
    async fn handle_message_full(&self, path: &str, msg: Message) -> anyhow::Result<Message> {
        self.dispatch_message(path, msg).await
    }

    /// Simple message handler for backward compatibility
    async fn handle_message_simple(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Message> {
        let msg = Message::single(msg_type, payload);
        self.dispatch_message(path, msg).await
    }

    async fn handle_tell(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()> {
        let msg = Message::single(msg_type, payload);
        self.dispatch_tell(path, msg).await
    }

    async fn handle_gossip(
        &self,
        payload: Vec<u8>,
        peer_addr: SocketAddr,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let cluster_guard = self.cluster.read().await;
        if let Some(backend) = cluster_guard.as_ref() {
            // Try to downcast to GossipBackend to access handle_gossip
            if let Some(gossip_backend) = backend.as_any().downcast_ref::<GossipBackend>() {
                let msg: GossipMessage = bincode::deserialize(&payload)?;
                let response = gossip_backend.inner().handle_gossip(msg, peer_addr).await?;
                if let Some(resp) = response {
                    Ok(Some(bincode::serialize(&resp)?))
                } else {
                    Ok(None)
                }
            } else {
                // Not a gossip backend, can't handle gossip messages
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    async fn health_check(&self) -> serde_json::Value {
        // Collect local actors info
        let mut actors = Vec::new();
        for entry in self.local_actors.iter() {
            let local_id = *entry.key();
            let handle = entry.value();

            // Find name from actor_names (reverse lookup)
            let name = self
                .actor_names
                .iter()
                .find(|e| *e.value() == local_id)
                .map(|e| e.key().clone())
                .unwrap_or_else(|| handle.actor_id.to_string());

            let mut actor_info = serde_json::json!({
                "name": name,
                "local_id": local_id,
                "stats": handle.stats.to_json(),
                "metadata": handle.metadata,
            });

            if let Some(path) = &handle.named_path {
                actor_info["named_path"] = serde_json::json!(path.as_str());
            }

            actors.push(actor_info);
        }

        // Collect named actors info
        let named_actors: Vec<_> = self
            .named_actor_paths
            .iter()
            .map(|e| {
                serde_json::json!({
                    "path": e.key().clone(),
                    "actor_name": e.value().clone(),
                })
            })
            .collect();

        // Collect cluster info
        let mut cluster_info = serde_json::json!(null);
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            let members = cluster.alive_members().await;
            let all_named = cluster.all_named_actors().await;

            cluster_info = serde_json::json!({
                "members_count": members.len(),
                "members": members,
                "named_actors_count": all_named.len(),
                "named_actors": all_named.iter().map(|info| {
                    serde_json::json!({
                        "path": info.path.as_str(),
                        "instance_count": info.instance_count(),
                    })
                }).collect::<Vec<_>>(),
            });
        }

        serde_json::json!({
            "node_id": self.node_id.to_string(),
            "actors_count": actors.len(),
            "actors": actors,
            "named_actors": named_actors,
            "cluster": cluster_info,
        })
    }

    async fn prometheus_metrics(&self) -> String {
        // Collect cluster member counts by status
        let mut cluster_members = std::collections::HashMap::new();
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            let all_members = cluster.all_members().await;
            for member in all_members {
                let status = format!("{:?}", member.status);
                *cluster_members.entry(status).or_insert(0usize) += 1;
            }
        }
        drop(cluster_guard);

        // Count messages from local actors
        let mut total_messages: u64 = 0;
        for entry in self.local_actors.iter() {
            total_messages += entry.value().stats.message_count.load(Ordering::Relaxed);
        }

        // Build system metrics
        let system_metrics = PrometheusMetrics {
            node_id: self.node_id.0,
            actors_count: self.local_actors.len(),
            messages_total: total_messages,
            actors_created: self.local_actors.len() as u64,
            actors_stopped: 0,
            cluster_members,
        };

        // Export using global metrics registry
        metrics().export_prometheus(&system_metrics)
    }

    async fn cluster_members(&self) -> serde_json::Value {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            let members = cluster.all_members().await;
            let result: Vec<_> = members
                .iter()
                .map(|m| {
                    serde_json::json!({
                        "node_id": m.node_id.to_string(),
                        "addr": m.addr.to_string(),
                        "status": format!("{:?}", m.status),
                    })
                })
                .collect();
            serde_json::json!(result)
        } else {
            serde_json::json!([{
                "node_id": self.node_id.to_string(),
                "status": "Alive",
            }])
        }
    }

    async fn actors_list(&self, include_internal: bool) -> serde_json::Value {
        let cluster_guard = self.cluster.read().await;
        let all_named = if let Some(cluster) = cluster_guard.as_ref() {
            cluster.all_named_actors().await
        } else {
            Vec::new()
        };
        drop(cluster_guard);

        // Build actors list with detailed info
        let mut actors = Vec::new();
        for info in all_named {
            let path_str = info.path.as_str();

            // Skip system/core
            if path_str == "system/core" {
                continue;
            }

            // Check if this actor is on this node
            if !info.instance_nodes.contains(&self.node_id) {
                continue;
            }

            let name = path_str.strip_prefix("actors/").unwrap_or(&path_str);

            // Skip internal actors unless requested
            if !include_internal && name.starts_with('_') {
                continue;
            }

            let actor_type = if name.starts_with('_') {
                "system"
            } else {
                "user"
            };

            // Get detailed instance info if available
            let mut actor_json = serde_json::json!({
                "name": name,
                "type": actor_type,
            });

            if let Some(instance) = info.get_instance(&self.node_id) {
                actor_json["actor_id"] = serde_json::json!(instance.actor_id.to_string());
                for (k, v) in &instance.metadata {
                    actor_json[k] = serde_json::json!(v);
                }
            }

            actors.push(actor_json);
        }

        serde_json::json!(actors)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn handle_head_api(
        &self,
        path: &str,
        method: &str,
        body: Vec<u8>,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        // Call the implementation method
        SystemMessageHandler::handle_head_api_impl(self, path, method, body).await
    }
}

impl SystemMessageHandler {
    // ========================================================================
    // Head Node API Handlers
    // ========================================================================

    /// Handle head node API requests (implementation)
    async fn handle_head_api_impl(
        &self,
        path: &str,
        method: &str,
        body: Vec<u8>,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let cluster_guard = self.cluster.read().await;
        let backend = cluster_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cluster backend not available"))?;

        // Try to downcast to HeadNodeBackend
        let head_backend = match backend.as_any().downcast_ref::<HeadNodeBackend>() {
            Some(b) if b.is_head() => b,
            _ => return Ok(None),
        };

        match (method, path) {
            ("POST", "/cluster/head/register") => {
                let req: RegisterNodeRequest = bincode::deserialize(&body)?;
                head_backend
                    .handle_register_node(req.node_id, req.addr)
                    .await?;
                Ok(Some(Vec::new()))
            }
            ("POST", "/cluster/head/heartbeat") => {
                let req: HeartbeatRequest = bincode::deserialize(&body)?;
                head_backend.handle_heartbeat(&req.node_id).await?;
                Ok(Some(Vec::new())) // Return empty body for success
            }
            ("POST", "/cluster/head/named_actor/register") => {
                let req: RegisterNamedActorRequest = bincode::deserialize(&body)?;
                head_backend
                    .handle_register_named_actor(req.path, req.node_id, req.actor_id, req.metadata)
                    .await?;
                Ok(Some(Vec::new())) // Return empty body for success
            }
            ("POST", "/cluster/head/named_actor/unregister") => {
                let req: UnregisterNamedActorRequest = bincode::deserialize(&body)?;
                head_backend
                    .handle_unregister_named_actor(&req.path, &req.node_id)
                    .await?;
                Ok(Some(Vec::new())) // Return empty body for success
            }
            ("GET", "/cluster/head/members") | ("POST", "/cluster/head/members") => {
                // Support both GET and POST (POST is used by Http2Client)
                let members = head_backend.all_members().await;
                let body = bincode::serialize(&members)?;
                Ok(Some(body))
            }
            ("GET", "/cluster/head/named_actors") | ("POST", "/cluster/head/named_actors") => {
                // Support both GET and POST (POST is used by Http2Client)
                let named_actors = head_backend.all_named_actors().await;
                let body = bincode::serialize(&named_actors)?;
                Ok(Some(body))
            }
            ("GET", "/cluster/head/sync") | ("POST", "/cluster/head/sync") => {
                let sync = head_backend.handle_sync().await?;
                let body = bincode::serialize(&sync)?;
                Ok(Some(body))
            }
            ("POST", "/cluster/head/actor/register") => {
                let req: RegisterActorRequest = bincode::deserialize(&body)?;
                head_backend
                    .handle_register_actor(req.actor_id, req.node_id)
                    .await?;
                Ok(Some(Vec::new()))
            }
            ("POST", "/cluster/head/actor/unregister") => {
                let req: UnregisterActorRequest = bincode::deserialize(&body)?;
                head_backend
                    .handle_unregister_actor(&req.actor_id, &req.node_id)
                    .await?;
                Ok(Some(Vec::new()))
            }
            _ => Err(anyhow::anyhow!(
                "Unknown head API endpoint: {} {}",
                method,
                path
            )),
        }
    }
}

// ============================================================================
// Head Node API Request Types
// ============================================================================

#[derive(Serialize, Deserialize)]
struct RegisterNodeRequest {
    node_id: NodeId,
    addr: SocketAddr,
}

#[derive(Serialize, Deserialize)]
struct HeartbeatRequest {
    node_id: NodeId,
}

#[derive(Serialize, Deserialize)]
struct RegisterNamedActorRequest {
    path: ActorPath,
    node_id: NodeId,
    actor_id: Option<ActorId>,
    metadata: HashMap<String, String>,
}

#[derive(Serialize, Deserialize)]
struct UnregisterNamedActorRequest {
    path: ActorPath,
    node_id: NodeId,
}
