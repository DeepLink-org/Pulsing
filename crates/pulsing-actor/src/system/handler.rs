//! HTTP/2 message handler for the actor system

use crate::actor::{ActorId, ActorPath, Envelope, Message, NodeId};
use crate::cluster::backends::{RegisterActorRequest, UnregisterActorRequest};
use crate::cluster::{GossipBackend, GossipMessage, HeadNodeBackend, NamingBackend};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::metrics::{metrics, SystemMetrics as PrometheusMetrics};
use crate::system::registry::ActorRegistry;
use crate::transport::Http2ServerHandler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Unified message handler for HTTP/2 transport.
///
/// Uses [`ActorRegistry`] for actor lookup instead of raw DashMaps.
pub(crate) struct SystemMessageHandler {
    node_id: NodeId,
    /// Actor registry for local actor management
    registry: Arc<ActorRegistry>,
    /// Cluster backend
    cluster: Arc<RwLock<Option<Arc<dyn NamingBackend>>>>,
}

impl SystemMessageHandler {
    pub fn new(
        node_id: NodeId,
        registry: Arc<ActorRegistry>,
        cluster: Arc<RwLock<Option<Arc<dyn NamingBackend>>>>,
    ) -> Self {
        Self {
            node_id,
            registry,
            cluster,
        }
    }

    /// Find actor sender by name or ActorId (O(1) lookup via registry)
    fn find_actor_sender(&self, actor_name: &str) -> Result<mpsc::Sender<Envelope>> {
        self.registry.find_actor_sender(actor_name).ok_or_else(|| {
            PulsingError::from(RuntimeError::actor_not_found(actor_name.to_string()))
        })
    }

    /// Dispatch a message to an actor (ask pattern)
    async fn dispatch_message(&self, path: &str, msg: Message) -> Result<Message> {
        if let Some(actor_name) = path.strip_prefix("/actors/") {
            self.send_to_local_actor(actor_name, msg).await
        } else if let Some(named_path) = path.strip_prefix("/named/") {
            self.send_to_named_actor(named_path, msg).await
        } else {
            Err(PulsingError::from(RuntimeError::invalid_actor_path(path)))
        }
    }

    /// Dispatch a fire-and-forget message
    async fn dispatch_tell(&self, path: &str, msg: Message) -> Result<()> {
        if let Some(actor_name) = path.strip_prefix("/actors/") {
            self.tell_local_actor(actor_name, msg).await
        } else if let Some(named_path) = path.strip_prefix("/named/") {
            self.tell_named_actor(named_path, msg).await
        } else {
            Err(PulsingError::from(RuntimeError::invalid_actor_path(path)))
        }
    }

    async fn send_to_local_actor(&self, actor_name: &str, msg: Message) -> Result<Message> {
        let sender = self.find_actor_sender(actor_name)?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let envelope = Envelope::ask(msg, tx);

        sender
            .send(envelope)
            .await
            .map_err(|_| PulsingError::from(RuntimeError::actor_stopped(actor_name)))?;

        rx.await
            .map_err(|_| PulsingError::from(RuntimeError::actor_stopped(actor_name)))?
    }

    async fn tell_local_actor(&self, actor_name: &str, msg: Message) -> Result<()> {
        let sender = self.find_actor_sender(actor_name)?;
        let envelope = Envelope::tell(msg);

        sender
            .send(envelope)
            .await
            .map_err(|_| PulsingError::from(RuntimeError::actor_stopped(actor_name)))?;

        Ok(())
    }

    async fn send_to_named_actor(&self, path: &str, msg: Message) -> Result<Message> {
        let actor_name = self
            .registry
            .get_actor_name_by_path(path)
            .ok_or_else(|| PulsingError::from(RuntimeError::named_actor_not_found(path)))?;

        self.send_to_local_actor(&actor_name, msg).await
    }

    async fn tell_named_actor(&self, path: &str, msg: Message) -> Result<()> {
        let actor_name = self
            .registry
            .get_actor_name_by_path(path)
            .ok_or_else(|| PulsingError::from(RuntimeError::named_actor_not_found(path)))?;

        self.tell_local_actor(&actor_name, msg).await
    }
}

#[async_trait::async_trait]
impl Http2ServerHandler for SystemMessageHandler {
    /// Unified message handler - accepts Message (Single or Stream), returns Message
    async fn handle_message_full(&self, path: &str, msg: Message) -> Result<Message> {
        self.dispatch_message(path, msg).await
    }

    /// Simple message handler for backward compatibility
    async fn handle_message_simple(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> Result<Message> {
        let msg = Message::single(msg_type, payload);
        self.dispatch_message(path, msg).await
    }

    async fn handle_tell(&self, path: &str, msg_type: &str, payload: Vec<u8>) -> Result<()> {
        let msg = Message::single(msg_type, payload);
        self.dispatch_tell(path, msg).await
    }

    async fn handle_gossip(
        &self,
        payload: Vec<u8>,
        peer_addr: SocketAddr,
    ) -> Result<Option<Vec<u8>>> {
        let cluster_guard = self.cluster.read().await;
        if let Some(backend) = cluster_guard.as_ref() {
            // Try to downcast to GossipBackend to access handle_gossip
            if let Some(gossip_backend) = backend.as_any().downcast_ref::<GossipBackend>() {
                let msg: GossipMessage = bincode::deserialize(&payload)
                    .map_err(|e| PulsingError::from(RuntimeError::Serialization(e.to_string())))?;
                let response = gossip_backend.inner().handle_gossip(msg, peer_addr).await?;
                if let Some(resp) = response {
                    Ok(Some(bincode::serialize(&resp).map_err(|e| {
                        PulsingError::from(RuntimeError::Serialization(e.to_string()))
                    })?))
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
        for entry in self.registry.iter_actors() {
            let local_id = *entry.key();
            let handle = entry.value();

            // Find name from actor_names (reverse lookup)
            let name = self
                .registry
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
            .registry
            .iter_named_paths()
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
        for entry in self.registry.iter_actors() {
            total_messages += entry.value().stats.message_count.load(Ordering::Relaxed);
        }

        // Build system metrics
        let system_metrics = PrometheusMetrics {
            node_id: self.node_id.0,
            actors_count: self.registry.actor_count(),
            messages_total: total_messages,
            actors_created: self.registry.actor_count() as u64,
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

    async fn handle_client_resolve(&self, path: &str) -> serde_json::Value {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            let actor_path = match ActorPath::new(path) {
                Ok(p) => p,
                Err(_) => {
                    return serde_json::json!({
                        "found": false,
                        "path": path,
                        "gateway": "",
                        "instance_count": 0,
                    });
                }
            };

            let info = cluster.lookup_named_actor(&actor_path).await;
            match info {
                Some(info) => {
                    let local_addr = cluster.local_addr();
                    serde_json::json!({
                        "found": true,
                        "path": path,
                        "gateway": local_addr.to_string(),
                        "instance_count": info.instance_count(),
                    })
                }
                None => {
                    serde_json::json!({
                        "found": false,
                        "path": path,
                        "gateway": "",
                        "instance_count": 0,
                    })
                }
            }
        } else {
            // Standalone mode: check local registry
            let found = self.registry.get_actor_name_by_path(path).is_some();
            serde_json::json!({
                "found": found,
                "path": path,
                "gateway": "",
                "instance_count": if found { 1 } else { 0 },
            })
        }
    }

    async fn handle_client_members(&self) -> serde_json::Value {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            let members = cluster.alive_members().await;
            let gateways: Vec<String> = members.iter().map(|m| m.addr.to_string()).collect();
            serde_json::json!({ "gateways": gateways })
        } else {
            serde_json::json!({ "gateways": [] })
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn handle_head_api(
        &self,
        path: &str,
        method: &str,
        body: Vec<u8>,
    ) -> Result<Option<Vec<u8>>> {
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
    ) -> Result<Option<Vec<u8>>> {
        let cluster_guard = self.cluster.read().await;
        let backend = cluster_guard
            .as_ref()
            .ok_or_else(|| PulsingError::from(RuntimeError::ClusterNotInitialized))?;

        let head_backend = match backend.as_any().downcast_ref::<HeadNodeBackend>() {
            Some(b) if b.is_head() => b,
            _ => return Ok(None),
        };

        let ser =
            |e: bincode::Error| PulsingError::from(RuntimeError::Serialization(e.to_string()));

        match (method, path) {
            ("POST", "/cluster/head/register") => {
                let req: RegisterNodeRequest = bincode::deserialize(&body).map_err(ser)?;
                head_backend
                    .handle_register_node(req.node_id, req.addr)
                    .await?;
                Ok(Some(Vec::new()))
            }
            ("POST", "/cluster/head/heartbeat") => {
                let req: HeartbeatRequest = bincode::deserialize(&body).map_err(ser)?;
                head_backend.handle_heartbeat(&req.node_id).await?;
                Ok(Some(Vec::new()))
            }
            ("POST", "/cluster/head/named_actor/register") => {
                let req: RegisterNamedActorRequest = bincode::deserialize(&body).map_err(ser)?;
                head_backend
                    .handle_register_named_actor(req.path, req.node_id, req.actor_id, req.metadata)
                    .await?;
                Ok(Some(Vec::new()))
            }
            ("POST", "/cluster/head/named_actor/unregister") => {
                let req: UnregisterNamedActorRequest = bincode::deserialize(&body).map_err(ser)?;
                head_backend
                    .handle_unregister_named_actor(&req.path, &req.node_id)
                    .await?;
                Ok(Some(Vec::new()))
            }
            ("GET", "/cluster/head/members") | ("POST", "/cluster/head/members") => {
                let members = head_backend.all_members().await;
                let body = bincode::serialize(&members).map_err(ser)?;
                Ok(Some(body))
            }
            ("GET", "/cluster/head/named_actors") | ("POST", "/cluster/head/named_actors") => {
                let named_actors = head_backend.all_named_actors().await;
                let body = bincode::serialize(&named_actors).map_err(ser)?;
                Ok(Some(body))
            }
            ("GET", "/cluster/head/sync") | ("POST", "/cluster/head/sync") => {
                let sync = head_backend.handle_sync().await?;
                let body = bincode::serialize(&sync).map_err(ser)?;
                Ok(Some(body))
            }
            ("POST", "/cluster/head/actor/register") => {
                let req: RegisterActorRequest = bincode::deserialize(&body).map_err(ser)?;
                head_backend
                    .handle_register_actor(req.actor_id, req.node_id)
                    .await?;
                Ok(Some(Vec::new()))
            }
            ("POST", "/cluster/head/actor/unregister") => {
                let req: UnregisterActorRequest = bincode::deserialize(&body).map_err(ser)?;
                head_backend
                    .handle_unregister_actor(&req.actor_id, &req.node_id)
                    .await?;
                Ok(Some(Vec::new()))
            }
            _ => Err(PulsingError::from(RuntimeError::Other(format!(
                "Unknown head API endpoint: {} {}",
                method, path
            )))),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::handle::{ActorStats, LocalActorHandle};
    use std::collections::HashMap;

    fn make_mock_actor(
        _actor_id: ActorId,
    ) -> (mpsc::Sender<Envelope>, tokio::task::JoinHandle<()>) {
        let (tx, mut rx) = mpsc::channel::<Envelope>(8);
        let join = tokio::spawn(async move {
            while let Some(env) = rx.recv().await {
                let (_, responder) = env.into_parts();
                responder.send(Ok(Message::single("pong", vec![1, 2, 3])));
            }
        });
        (tx, join)
    }

    #[tokio::test]
    async fn test_dispatch_message_invalid_path() {
        let registry = Arc::new(ActorRegistry::new());
        let cluster = Arc::new(RwLock::new(None));
        let handler = SystemMessageHandler::new(NodeId::generate(), registry, cluster);
        let msg = Message::single("ping", vec![]);

        let err = handler
            .handle_message_full("/invalid/path", msg)
            .await
            .unwrap_err();
        assert!(err.is_runtime());
        assert!(err.to_string().to_lowercase().contains("path"));
    }

    #[tokio::test]
    async fn test_dispatch_message_actor_not_found() {
        let registry = Arc::new(ActorRegistry::new());
        let cluster = Arc::new(RwLock::new(None));
        let handler = SystemMessageHandler::new(NodeId::generate(), registry, cluster);
        let msg = Message::single("ping", vec![]);

        let err = handler
            .handle_message_full("/actors/missing-actor", msg)
            .await
            .unwrap_err();
        assert!(err.is_runtime());
        assert!(err.to_string().contains("missing-actor"));
    }

    #[tokio::test]
    async fn test_dispatch_message_named_actor_not_found() {
        let registry = Arc::new(ActorRegistry::new());
        let cluster = Arc::new(RwLock::new(None));
        let handler = SystemMessageHandler::new(NodeId::generate(), registry, cluster);
        let msg = Message::single("ping", vec![]);

        let err = handler
            .handle_message_full("/named/services/foo", msg)
            .await
            .unwrap_err();
        assert!(err.is_runtime());
        assert!(
            err.to_string().contains("named_actor_not_found")
                || err.to_string().contains("services/foo")
        );
    }

    #[tokio::test]
    async fn test_dispatch_message_success() {
        let actor_id = ActorId::generate();
        let (sender, join_handle) = make_mock_actor(actor_id);
        let handle = LocalActorHandle {
            sender,
            join_handle,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            stats: Arc::new(ActorStats::default()),
            metadata: HashMap::new(),
            named_path: None,
            actor_id,
        };

        let registry = Arc::new(ActorRegistry::new());
        registry.register_actor(actor_id, handle);
        registry.register_name("test-actor".to_string(), actor_id);

        let cluster = Arc::new(RwLock::new(None));
        let handler = SystemMessageHandler::new(NodeId::generate(), registry, cluster);
        let msg = Message::single("ping", vec![]);

        let response = handler
            .handle_message_full("/actors/test-actor", msg)
            .await
            .unwrap();
        assert_eq!(response.msg_type(), "pong");
    }

    #[tokio::test]
    async fn test_handle_message_simple_routing() {
        let actor_id = ActorId::generate();
        let (sender, join_handle) = make_mock_actor(actor_id);
        let handle = LocalActorHandle {
            sender,
            join_handle,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            stats: Arc::new(ActorStats::default()),
            metadata: HashMap::new(),
            named_path: None,
            actor_id,
        };

        let registry = Arc::new(ActorRegistry::new());
        registry.register_actor(actor_id, handle);
        registry.register_name("simple".to_string(), actor_id);

        let cluster = Arc::new(RwLock::new(None));
        let handler = SystemMessageHandler::new(NodeId::generate(), registry, cluster);

        let response = handler
            .handle_message_simple("/actors/simple", "req", vec![1, 2, 3])
            .await
            .unwrap();
        assert_eq!(response.msg_type(), "pong");
    }

    #[tokio::test]
    async fn test_dispatch_tell_actor_not_found() {
        let registry = Arc::new(ActorRegistry::new());
        let cluster = Arc::new(RwLock::new(None));
        let handler = SystemMessageHandler::new(NodeId::generate(), registry, cluster);

        let err = handler
            .handle_tell("/actors/missing", "msg", vec![])
            .await
            .unwrap_err();
        assert!(err.is_runtime());
    }
}
