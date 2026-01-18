//! Gossip backend implementation
//!
//! Wraps the existing GossipCluster to implement the NamingBackend trait.

use crate::actor::{ActorId, ActorPath, NodeId, StopReason};
use crate::cluster::NamingBackend;
use crate::cluster::{
    member::{MemberInfo, NamedActorInfo, NamedActorInstance},
    GossipCluster, GossipConfig,
};
use crate::transport::http2::Http2Transport;
use async_trait::async_trait;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Gossip-based naming backend
///
/// This wraps the existing GossipCluster implementation to provide
/// the NamingBackend trait interface.
pub struct GossipBackend {
    cluster: Arc<GossipCluster>,
}

impl GossipBackend {
    /// Create a new GossipBackend
    pub fn new(
        local_node: NodeId,
        local_addr: SocketAddr,
        transport: Arc<Http2Transport>,
        config: GossipConfig,
    ) -> Self {
        let cluster = GossipCluster::new(local_node, local_addr, transport, config);
        Self {
            cluster: Arc::new(cluster),
        }
    }

    /// Get a reference to the inner GossipCluster
    ///
    /// This is needed for SystemMessageHandler to access handle_gossip method.
    pub fn inner(&self) -> &GossipCluster {
        &self.cluster
    }
}

#[async_trait]
impl NamingBackend for GossipBackend {
    // ========================================================================
    // Node Management
    // ========================================================================

    async fn join(&self, seeds: Vec<SocketAddr>) -> anyhow::Result<()> {
        self.cluster.join(seeds).await
    }

    async fn leave(&self) -> anyhow::Result<()> {
        self.cluster.leave().await
    }

    async fn all_members(&self) -> Vec<MemberInfo> {
        self.cluster.all_members().await
    }

    async fn alive_members(&self) -> Vec<MemberInfo> {
        self.cluster.alive_members().await
    }

    async fn get_member(&self, node_id: &NodeId) -> Option<MemberInfo> {
        self.cluster.get_member(node_id).await
    }

    // ========================================================================
    // Named Actor Registration
    // ========================================================================

    async fn register_named_actor(&self, path: ActorPath) {
        self.cluster.register_named_actor(path).await
    }

    async fn register_named_actor_full(
        &self,
        path: ActorPath,
        actor_id: ActorId,
        metadata: HashMap<String, String>,
    ) {
        self.cluster
            .register_named_actor_full(path, actor_id, metadata)
            .await
    }

    async fn unregister_named_actor(&self, path: &ActorPath) {
        self.cluster.unregister_named_actor(path).await
    }

    async fn broadcast_named_actor_failed(&self, path: &ActorPath, reason: &StopReason) {
        self.cluster
            .broadcast_named_actor_failed(path, reason)
            .await
    }

    // ========================================================================
    // Named Actor Queries
    // ========================================================================

    async fn lookup_named_actor(&self, path: &ActorPath) -> Option<NamedActorInfo> {
        self.cluster.lookup_named_actor(path).await
    }

    async fn select_named_actor_instance(&self, path: &ActorPath) -> Option<MemberInfo> {
        self.cluster.select_named_actor_instance(path).await
    }

    async fn get_named_actor_instances(&self, path: &ActorPath) -> Vec<MemberInfo> {
        self.cluster.get_named_actor_instances(path).await
    }

    async fn get_named_actor_instances_detailed(
        &self,
        path: &ActorPath,
    ) -> Vec<(MemberInfo, Option<NamedActorInstance>)> {
        self.cluster.get_named_actor_instances_detailed(path).await
    }

    async fn all_named_actors(&self) -> Vec<NamedActorInfo> {
        self.cluster.all_named_actors().await
    }

    // ========================================================================
    // Actor Registration
    // ========================================================================

    async fn register_actor(&self, actor_id: ActorId) {
        self.cluster.register_actor(actor_id).await
    }

    async fn unregister_actor(&self, actor_id: &ActorId) {
        self.cluster.unregister_actor(actor_id).await
    }

    async fn lookup_actor(&self, actor_id: &ActorId) -> Option<MemberInfo> {
        self.cluster.lookup_actor(actor_id).await
    }

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    fn start(&self, cancel: CancellationToken) {
        self.cluster.start(cancel)
    }

    fn local_node(&self) -> &NodeId {
        self.cluster.local_node()
    }

    fn local_addr(&self) -> SocketAddr {
        self.cluster.local_addr()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
