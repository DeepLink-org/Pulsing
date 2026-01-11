//! Gossip protocol for cluster membership and actor discovery
//!
//! Implements a Redis Cluster-style gossip protocol with:
//! - MEET/PING/PONG message exchange
//! - Configuration epoch for conflict resolution
//! - Partial view propagation to reduce message size
//! - PFail/Fail failure detection

use super::member::{
    ActorLocation, ClusterNode, FailureInfo, MemberInfo, MemberStatus, NamedActorInfo,
    NamedActorInstance, NodeStatus,
};
use super::swim::SwimConfig;
use crate::actor::{ActorId, ActorPath, NodeId, StopReason};
use crate::transport::http2::Http2Transport;
use rand::prelude::IndexedRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Utility Functions
// ============================================================================

/// Get current timestamp in milliseconds since UNIX epoch
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Fix 0.0.0.0 addresses using peer's actual IP
fn fix_addr(addr: SocketAddr, peer_ip: std::net::IpAddr) -> SocketAddr {
    if addr.ip().is_unspecified() {
        SocketAddr::new(peer_ip, addr.port())
    } else {
        addr
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Gossip protocol configuration
#[derive(Clone, Debug)]
pub struct GossipConfig {
    /// Interval between gossip rounds
    pub gossip_interval: Duration,
    /// Number of nodes to gossip with per round (fanout)
    pub fanout: usize,
    /// Number of times to probe each seed node on startup
    pub seed_probe_count: usize,
    /// Delay between seed probes
    pub seed_probe_interval: Duration,
    /// Interval for periodic seed re-probing (None to disable)
    pub seed_rejoin_interval: Option<Duration>,
    /// Timeout before marking a node as PFail
    pub failure_timeout: Duration,
    /// Grace period before removing Fail nodes
    pub cleanup_grace_period: Duration,
    /// SWIM config (for ping interval and suspicion timeout)
    pub swim: SwimConfig,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            gossip_interval: Duration::from_millis(200),
            fanout: 3,
            seed_probe_count: 3,
            seed_probe_interval: Duration::from_millis(100),
            seed_rejoin_interval: Some(Duration::from_secs(15)),
            // Increased from 5s to 15s for better tolerance in high-load scenarios
            // In large-scale stress tests, gossip messages may be delayed due to high load
            failure_timeout: Duration::from_secs(15),
            cleanup_grace_period: Duration::from_secs(30),
            swim: SwimConfig::default(),
        }
    }
}

// ============================================================================
// Messages
// ============================================================================

/// Gossip protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipMessage {
    /// MEET: Invite a new node to join the cluster
    Meet {
        from: NodeId,
        from_addr: SocketAddr,
        current_epoch: u64,
    },

    /// PING: Periodic probe with partial cluster view
    Ping {
        from: NodeId,
        current_epoch: u64,
        partial_view: Vec<ClusterNode>,
        failures: Vec<FailureInfo>,
        named_actors: Option<Vec<NamedActorInfo>>,
    },

    /// PONG: Response to PING/MEET
    Pong {
        from: NodeId,
        current_epoch: u64,
        partial_view: Vec<ClusterNode>,
        failures: Vec<FailureInfo>,
        named_actors: Option<Vec<NamedActorInfo>>,
    },

    // Legacy actor messages (kept for compatibility)
    ActorRegistered {
        location: ActorLocation,
    },
    ActorUnregistered {
        actor_id: ActorId,
    },
    NamedActorRegistered {
        path: ActorPath,
        node_id: NodeId,
        /// Actor ID (optional for backward compatibility)
        #[serde(default)]
        actor_id: Option<ActorId>,
        /// Metadata (e.g., Python class, module, file path)
        #[serde(default)]
        metadata: std::collections::HashMap<String, String>,
    },
    NamedActorUnregistered {
        path: ActorPath,
        node_id: NodeId,
    },
    NamedActorFailed {
        path: ActorPath,
        node_id: NodeId,
        reason: String,
    },
}

// ============================================================================
// Shared State
// ============================================================================

/// Shared cluster state (used by both GossipCluster and background tasks)
struct ClusterState {
    local_node: NodeId,
    local_addr: SocketAddr,
    cluster_nodes: RwLock<HashMap<NodeId, ClusterNode>>,
    named_actors: RwLock<HashMap<String, NamedActorInfo>>,
    actors: RwLock<HashMap<ActorId, NodeId>>,
    seed_addrs: RwLock<Vec<SocketAddr>>,
    current_epoch: AtomicU64,
}

impl ClusterState {
    fn new(local_node: NodeId, local_addr: SocketAddr) -> Self {
        let mut cluster_nodes = HashMap::new();
        cluster_nodes.insert(local_node, ClusterNode::new(local_node, local_addr, 0));

        Self {
            local_node,
            local_addr,
            cluster_nodes: RwLock::new(cluster_nodes),
            named_actors: RwLock::new(HashMap::new()),
            actors: RwLock::new(HashMap::new()),
            seed_addrs: RwLock::new(Vec::new()),
            current_epoch: AtomicU64::new(0),
        }
    }

    fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::SeqCst)
    }

    fn increment_epoch(&self) -> u64 {
        self.current_epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get partial cluster view for gossip
    async fn get_partial_view(&self, count: usize) -> Vec<ClusterNode> {
        let nodes = self.cluster_nodes.read().await;
        let mut view: Vec<_> = nodes
            .values()
            .filter(|n| n.status != NodeStatus::Fail)
            .cloned()
            .collect();
        drop(nodes);

        use rand::prelude::SliceRandom;
        let mut rng = rand::rng();
        view.shuffle(&mut rng);
        view.into_iter().take(count).collect()
    }

    /// Get failure information for gossip
    async fn get_failures(&self) -> Vec<FailureInfo> {
        let nodes = self.cluster_nodes.read().await;
        nodes
            .values()
            .filter(|n| n.status == NodeStatus::PFail || n.status == NodeStatus::Fail)
            .map(|n| FailureInfo {
                node_id: n.node_id,
                status: n.status,
                epoch: n.epoch,
                reported_by: self.local_node,
            })
            .collect()
    }

    /// Get partial named actors for gossip
    async fn get_partial_named_actors(&self, count: usize) -> Vec<NamedActorInfo> {
        let named_actors = self.named_actors.read().await;
        let mut actors: Vec<_> = named_actors.values().cloned().collect();
        drop(named_actors);

        use rand::prelude::SliceRandom;
        let mut rng = rand::rng();
        actors.shuffle(&mut rng);
        actors.into_iter().take(count).collect()
    }

    /// Merge cluster node info using epoch for conflict resolution
    async fn merge_node(&self, remote: &ClusterNode, peer_addr: SocketAddr) {
        let fixed_addr = fix_addr(remote.addr, peer_addr.ip());
        let mut nodes = self.cluster_nodes.write().await;

        match nodes.get(&remote.node_id) {
            Some(existing) => {
                if remote.epoch > existing.epoch
                    || (remote.epoch == existing.epoch && remote.status > existing.status)
                {
                    nodes.insert(
                        remote.node_id,
                        ClusterNode {
                            node_id: remote.node_id,
                            addr: fixed_addr,
                            status: remote.status,
                            epoch: remote.epoch,
                            last_seen: remote.last_seen,
                        },
                    );
                }
            }
            None => {
                nodes.insert(
                    remote.node_id,
                    ClusterNode {
                        node_id: remote.node_id,
                        addr: fixed_addr,
                        status: remote.status,
                        epoch: remote.epoch,
                        last_seen: remote.last_seen,
                    },
                );
                tracing::debug!(node_id = %remote.node_id, "Added new node from gossip");
            }
        }
    }

    /// Handle failure info from gossip
    async fn handle_failure(&self, failure: &FailureInfo) {
        let mut nodes = self.cluster_nodes.write().await;
        if let Some(node) = nodes.get_mut(&failure.node_id) {
            if failure.epoch >= node.epoch && failure.status > node.status {
                node.status = failure.status;
                node.epoch = failure.epoch;
            }
        }
    }

    /// Merge named actors from gossip
    async fn merge_named_actors(&self, remote: Vec<NamedActorInfo>) {
        let mut local = self.named_actors.write().await;
        for info in remote {
            let key = info.path.as_str();
            if let Some(existing) = local.get_mut(&key) {
                existing.merge(&info);
            } else {
                local.insert(key, info);
            }
        }
    }

    /// Update node's last_seen and recover from PFail if active
    async fn touch_node(&self, node_id: NodeId) {
        let mut nodes = self.cluster_nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            node.last_seen = now_millis();
            if node.status == NodeStatus::PFail {
                node.status = NodeStatus::Online;
                node.epoch = self.increment_epoch();
                tracing::info!(node_id = %node_id, "Node recovered from PFail");
            }
        }
    }

    /// Convert ClusterNode to MemberInfo for API compatibility
    fn node_to_member(node: &ClusterNode) -> MemberInfo {
        let status = match node.status {
            NodeStatus::Online | NodeStatus::Handshake => MemberStatus::Alive,
            NodeStatus::PFail => MemberStatus::Suspect,
            NodeStatus::Fail => MemberStatus::Dead,
        };
        MemberInfo {
            node_id: node.node_id,
            addr: node.addr,
            gossip_addr: node.addr,
            status,
            incarnation: node.epoch,
            last_update: None,
        }
    }
}

// ============================================================================
// GossipCluster
// ============================================================================

/// Gossip cluster manager
pub struct GossipCluster {
    state: Arc<ClusterState>,
    transport: Arc<Http2Transport>,
    config: GossipConfig,
}

impl GossipCluster {
    pub fn new(
        local_node: NodeId,
        local_addr: SocketAddr,
        transport: Arc<Http2Transport>,
        config: GossipConfig,
    ) -> Self {
        tracing::info!(node_id = %local_node, addr = %local_addr, "Starting gossip cluster");

        Self {
            state: Arc::new(ClusterState::new(local_node, local_addr)),
            transport,
            config,
        }
    }

    // ========================================================================
    // Public API
    // ========================================================================

    pub fn local_node(&self) -> &NodeId {
        &self.state.local_node
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.state.local_addr
    }

    pub fn current_epoch(&self) -> u64 {
        self.state.current_epoch()
    }

    pub fn increment_epoch(&self) -> u64 {
        self.state.increment_epoch()
    }

    /// Join cluster via seed nodes
    pub async fn join(&self, seed_addrs: Vec<SocketAddr>) -> anyhow::Result<()> {
        if seed_addrs.is_empty() {
            tracing::info!("No seed nodes provided, starting as first node");
            return Ok(());
        }

        *self.state.seed_addrs.write().await = seed_addrs.clone();

        let msg = GossipMessage::Meet {
            from: self.state.local_node,
            from_addr: self.state.local_addr,
            current_epoch: self.state.current_epoch(),
        };
        let payload = bincode::serialize(&msg)?;

        tracing::info!(seeds = ?seed_addrs, "Probing seed nodes");

        for probe in 0..self.config.seed_probe_count.max(1) {
            for addr in &seed_addrs {
                if let Ok(Some(resp)) = self.transport.send_gossip(*addr, payload.clone()).await {
                    if let Ok(msg) = bincode::deserialize(&resp) {
                        let _ = self.handle_gossip(msg, *addr).await;
                    }
                }
            }

            if self.state.cluster_nodes.read().await.len() >= self.config.fanout {
                break;
            }

            if probe < self.config.seed_probe_count - 1 {
                tokio::time::sleep(self.config.seed_probe_interval).await;
            }
        }

        let node_count = self.state.cluster_nodes.read().await.len();
        tracing::info!(nodes = node_count, "Seed probing complete");
        Ok(())
    }

    /// Start background gossip loops
    pub fn start(&self, cancel: CancellationToken) {
        let state = self.state.clone();
        let transport = self.transport.clone();
        let config = self.config.clone();
        let cancel2 = cancel.clone();
        tokio::spawn(async move {
            gossip_loop(state, transport, config, cancel2).await;
        });

        let state = self.state.clone();
        let config = self.config.clone();
        let cancel2 = cancel.clone();
        tokio::spawn(async move {
            failure_detection_loop(state, config, cancel2).await;
        });

        if let Some(interval) = self.config.seed_rejoin_interval {
            let state = self.state.clone();
            let transport = self.transport.clone();
            let config = self.config.clone();
            tokio::spawn(async move {
                seed_rejoin_loop(state, transport, config, interval, cancel).await;
            });
        }
    }

    /// Leave cluster gracefully
    pub async fn leave(&self) -> anyhow::Result<()> {
        tracing::info!(node_id = %self.state.local_node, "Leaving cluster");
        Ok(())
    }

    /// Handle incoming gossip message
    pub async fn handle_gossip(
        &self,
        msg: GossipMessage,
        peer_addr: SocketAddr,
    ) -> anyhow::Result<Option<GossipMessage>> {
        match msg {
            GossipMessage::Meet {
                from,
                from_addr,
                current_epoch,
            } => {
                let fixed_addr = fix_addr(from_addr, peer_addr.ip());

                let mut nodes = self.state.cluster_nodes.write().await;
                nodes
                    .entry(from)
                    .and_modify(|n| {
                        n.addr = fixed_addr;
                        n.last_seen = now_millis();
                        if n.epoch < current_epoch {
                            n.epoch = current_epoch;
                        }
                        if n.status != NodeStatus::Online {
                            n.status = NodeStatus::Online;
                        }
                    })
                    .or_insert_with(|| ClusterNode {
                        node_id: from,
                        addr: fixed_addr,
                        status: NodeStatus::Online,
                        epoch: current_epoch,
                        last_seen: now_millis(),
                    });
                drop(nodes);

                // Return full cluster info to new node
                let all_named = self
                    .state
                    .named_actors
                    .read()
                    .await
                    .values()
                    .cloned()
                    .collect();
                Ok(Some(GossipMessage::Pong {
                    from: self.state.local_node,
                    current_epoch: self.state.current_epoch(),
                    partial_view: self.state.get_partial_view(10).await,
                    failures: self.state.get_failures().await,
                    named_actors: Some(all_named),
                }))
            }

            GossipMessage::Ping {
                from,
                current_epoch: _,
                partial_view,
                failures,
                named_actors,
            } => {
                self.state.touch_node(from).await;

                for node in &partial_view {
                    self.state.merge_node(node, peer_addr).await;
                }
                for f in &failures {
                    self.state.handle_failure(f).await;
                }
                if let Some(actors) = named_actors {
                    self.state.merge_named_actors(actors).await;
                }

                Ok(Some(GossipMessage::Pong {
                    from: self.state.local_node,
                    current_epoch: self.state.current_epoch(),
                    partial_view: self.state.get_partial_view(10).await,
                    failures: self.state.get_failures().await,
                    named_actors: Some(self.state.get_partial_named_actors(10).await),
                }))
            }

            GossipMessage::Pong {
                from,
                current_epoch: _,
                partial_view,
                failures,
                named_actors,
            } => {
                self.state.touch_node(from).await;

                for node in &partial_view {
                    self.state.merge_node(node, peer_addr).await;
                }
                for f in &failures {
                    self.state.handle_failure(f).await;
                }
                if let Some(actors) = named_actors {
                    self.state.merge_named_actors(actors).await;
                }

                Ok(None)
            }

            // Legacy messages
            GossipMessage::ActorRegistered { location } => {
                self.state
                    .actors
                    .write()
                    .await
                    .insert(location.actor_id, location.node_id);
                Ok(None)
            }
            GossipMessage::ActorUnregistered { actor_id } => {
                self.state.actors.write().await.remove(&actor_id);
                Ok(None)
            }
            GossipMessage::NamedActorRegistered {
                path,
                node_id,
                actor_id,
                metadata,
            } => {
                let mut named = self.state.named_actors.write().await;
                let key = path.as_str();
                if let Some(aid) = actor_id {
                    // Full registration with actor_id and metadata
                    let instance = NamedActorInstance::with_metadata(node_id, aid, metadata);
                    named
                        .entry(key.clone())
                        .and_modify(|info| info.add_full_instance(instance.clone()))
                        .or_insert_with(|| NamedActorInfo::with_full_instance(path, instance));
                } else {
                    // Legacy registration (no actor_id)
                    named
                        .entry(key.clone())
                        .and_modify(|info| info.add_instance(node_id))
                        .or_insert_with(|| NamedActorInfo::with_instance(path, node_id));
                }
                Ok(None)
            }
            GossipMessage::NamedActorUnregistered { path, node_id } => {
                let mut named = self.state.named_actors.write().await;
                let key = path.as_str();
                if let Some(info) = named.get_mut(&key) {
                    info.remove_instance(&node_id);
                    if info.is_empty() {
                        named.remove(&key);
                    }
                }
                Ok(None)
            }
            GossipMessage::NamedActorFailed {
                path,
                node_id,
                reason,
            } => {
                tracing::warn!(path = %path, node_id = %node_id, reason = %reason, "Named actor failed");
                let mut named = self.state.named_actors.write().await;
                let key = path.as_str();
                if let Some(info) = named.get_mut(&key) {
                    info.remove_instance(&node_id);
                    if info.is_empty() {
                        named.remove(&key);
                    }
                }
                Ok(None)
            }
        }
    }

    // ========================================================================
    // Actor Registry
    // ========================================================================

    pub async fn register_actor(&self, actor_id: ActorId) {
        self.state
            .actors
            .write()
            .await
            .insert(actor_id, self.state.local_node);
        let msg = GossipMessage::ActorRegistered {
            location: ActorLocation::new(actor_id, self.state.local_node),
        };
        let _ = self.broadcast(&msg).await;
    }

    pub async fn unregister_actor(&self, actor_id: &ActorId) {
        self.state.actors.write().await.remove(actor_id);
        let msg = GossipMessage::ActorUnregistered {
            actor_id: *actor_id,
        };
        let _ = self.broadcast(&msg).await;
    }

    pub async fn lookup_actor(&self, actor_id: &ActorId) -> Option<MemberInfo> {
        let node_id = self.state.actors.read().await.get(actor_id).copied()?;
        self.get_member(&node_id).await
    }

    // ========================================================================
    // Named Actor Registry
    // ========================================================================

    /// Register a named actor with full details (actor_id and metadata)
    pub async fn register_named_actor_full(
        &self,
        path: ActorPath,
        actor_id: ActorId,
        metadata: std::collections::HashMap<String, String>,
    ) {
        let key = path.as_str();
        let instance =
            NamedActorInstance::with_metadata(self.state.local_node, actor_id, metadata.clone());
        {
            let mut named = self.state.named_actors.write().await;
            named
                .entry(key.clone())
                .and_modify(|info| info.add_full_instance(instance.clone()))
                .or_insert_with(|| NamedActorInfo::with_full_instance(path.clone(), instance));
        }
        let msg = GossipMessage::NamedActorRegistered {
            path,
            node_id: self.state.local_node,
            actor_id: Some(actor_id),
            metadata,
        };
        let _ = self.broadcast(&msg).await;
    }

    /// Register a named actor (legacy, without actor_id)
    pub async fn register_named_actor(&self, path: ActorPath) {
        let key = path.as_str();
        {
            let mut named = self.state.named_actors.write().await;
            named
                .entry(key.clone())
                .and_modify(|info| info.add_instance(self.state.local_node))
                .or_insert_with(|| {
                    NamedActorInfo::with_instance(path.clone(), self.state.local_node)
                });
        }
        let msg = GossipMessage::NamedActorRegistered {
            path,
            node_id: self.state.local_node,
            actor_id: None,
            metadata: std::collections::HashMap::new(),
        };
        let _ = self.broadcast(&msg).await;
    }

    pub async fn unregister_named_actor(&self, path: &ActorPath) {
        let key = path.as_str();
        {
            let mut named = self.state.named_actors.write().await;
            if let Some(info) = named.get_mut(&key) {
                info.remove_instance(&self.state.local_node);
                if info.is_empty() {
                    named.remove(&key);
                }
            }
        }
        let msg = GossipMessage::NamedActorUnregistered {
            path: path.clone(),
            node_id: self.state.local_node,
        };
        let _ = self.broadcast(&msg).await;
    }

    pub async fn broadcast_named_actor_failed(&self, path: &ActorPath, reason: &StopReason) {
        let msg = GossipMessage::NamedActorFailed {
            path: path.clone(),
            node_id: self.state.local_node,
            reason: reason.to_string(),
        };
        let _ = self.broadcast(&msg).await;
    }

    pub async fn lookup_named_actor(&self, path: &ActorPath) -> Option<NamedActorInfo> {
        self.state
            .named_actors
            .read()
            .await
            .get(&path.as_str())
            .cloned()
    }

    pub async fn select_named_actor_instance(&self, path: &ActorPath) -> Option<MemberInfo> {
        let node_id = {
            let named = self.state.named_actors.read().await;
            named.get(&path.as_str())?.select_instance()?
        };
        self.get_member(&node_id).await
    }

    pub async fn get_named_actor_instances(&self, path: &ActorPath) -> Vec<MemberInfo> {
        let node_ids = {
            let named = self.state.named_actors.read().await;
            match named.get(&path.as_str()) {
                Some(info) => info.instance_nodes.clone(),
                None => return Vec::new(),
            }
        };

        let nodes = self.state.cluster_nodes.read().await;
        node_ids
            .iter()
            .filter_map(|id| nodes.get(id).map(ClusterState::node_to_member))
            .collect()
    }

    /// Get detailed instance information for a named actor
    pub async fn get_named_actor_instances_detailed(
        &self,
        path: &ActorPath,
    ) -> Vec<(MemberInfo, Option<NamedActorInstance>)> {
        let (node_ids, instances_map) = {
            let named = self.state.named_actors.read().await;
            match named.get(&path.as_str()) {
                Some(info) => (info.instance_nodes.clone(), info.instances.clone()),
                None => return Vec::new(),
            }
        };

        let nodes = self.state.cluster_nodes.read().await;
        node_ids
            .iter()
            .filter_map(|id| {
                nodes.get(id).map(|node| {
                    let member = ClusterState::node_to_member(node);
                    let instance = instances_map.get(id).cloned();
                    (member, instance)
                })
            })
            .collect()
    }

    pub async fn all_named_actors(&self) -> Vec<NamedActorInfo> {
        self.state
            .named_actors
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    // ========================================================================
    // Member Queries
    // ========================================================================

    pub async fn all_members(&self) -> Vec<MemberInfo> {
        self.state
            .cluster_nodes
            .read()
            .await
            .values()
            .map(ClusterState::node_to_member)
            .collect()
    }

    pub async fn alive_members(&self) -> Vec<MemberInfo> {
        self.state
            .cluster_nodes
            .read()
            .await
            .values()
            .filter(|n| n.status == NodeStatus::Online && n.node_id != self.state.local_node)
            .map(ClusterState::node_to_member)
            .collect()
    }

    pub async fn get_member(&self, node_id: &NodeId) -> Option<MemberInfo> {
        self.state
            .cluster_nodes
            .read()
            .await
            .get(node_id)
            .map(ClusterState::node_to_member)
    }

    // ========================================================================
    // Legacy API compatibility
    // ========================================================================

    pub async fn get_partial_view(&self, count: usize) -> Vec<ClusterNode> {
        self.state.get_partial_view(count).await
    }

    pub async fn get_failures_to_propagate(&self) -> Vec<FailureInfo> {
        self.state.get_failures().await
    }

    pub async fn get_partial_named_actors(&self, count: usize) -> Vec<NamedActorInfo> {
        self.state.get_partial_named_actors(count).await
    }

    pub async fn merge_cluster_node(&self, remote: &ClusterNode, peer_addr: SocketAddr) {
        self.state.merge_node(remote, peer_addr).await;
    }

    pub async fn handle_failure_info(&self, failure: &FailureInfo) {
        self.state.handle_failure(failure).await;
    }

    pub async fn update_node_epoch(&self, _node_id: NodeId, _epoch: u64) -> bool {
        // Simplified - epoch tracking is done in merge_node
        true
    }

    pub async fn mark_pfail(&self, node_id: NodeId) {
        let mut nodes = self.state.cluster_nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            if node.status == NodeStatus::Online {
                node.status = NodeStatus::PFail;
                node.epoch = self.state.increment_epoch();
                tracing::warn!(node_id = %node_id, "Marked node as PFail");
            }
        }
    }

    pub async fn mark_fail(&self, node_id: NodeId) {
        let mut nodes = self.state.cluster_nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            if node.status != NodeStatus::Fail {
                node.status = NodeStatus::Fail;
                node.epoch = self.state.increment_epoch();
                tracing::error!(node_id = %node_id, "Marked node as Fail");
            }
        }
    }

    pub async fn check_pfail_to_fail(&self) {
        // Simplified: check if PFail nodes should become Fail
        let pfail_nodes: Vec<_> = {
            let nodes = self.state.cluster_nodes.read().await;
            nodes
                .values()
                .filter(|n| n.status == NodeStatus::PFail)
                .map(|n| (n.node_id, n.last_seen))
                .collect()
        };

        let timeout_ms = self.config.swim.suspicion_timeout.as_millis() as u64;
        let now = now_millis();

        for (node_id, last_seen) in pfail_nodes {
            if now.saturating_sub(last_seen) > timeout_ms {
                self.mark_fail(node_id).await;
            }
        }
    }

    pub async fn merge_named_actors(&self, actors: Vec<NamedActorInfo>) {
        self.state.merge_named_actors(actors).await;
    }

    // ========================================================================
    // Internal
    // ========================================================================

    async fn broadcast(&self, msg: &GossipMessage) -> anyhow::Result<()> {
        let payload = bincode::serialize(msg)?;
        let members = self.alive_members().await;

        let targets: Vec<_> = {
            let mut rng = rand::rng();
            members
                .choose_multiple(&mut rng, self.config.fanout.min(members.len()))
                .cloned()
                .collect()
        };

        for m in targets {
            let _ = self.transport.send_gossip(m.addr, payload.clone()).await;
        }
        Ok(())
    }
}

// ============================================================================
// Background Tasks
// ============================================================================

async fn gossip_loop(
    state: Arc<ClusterState>,
    transport: Arc<Http2Transport>,
    config: GossipConfig,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(config.gossip_interval);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                gossip_round(&state, &transport, &config).await;
            }
            _ = cancel.cancelled() => {
                tracing::info!("Gossip loop shutting down");
                break;
            }
        }
    }
}

async fn gossip_round(state: &ClusterState, transport: &Http2Transport, config: &GossipConfig) {
    let targets: Vec<ClusterNode> = {
        let nodes = state.cluster_nodes.read().await;
        let mut online: Vec<_> = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Online && n.node_id != state.local_node)
            .cloned()
            .collect();

        if online.is_empty() {
            return;
        }

        use rand::prelude::SliceRandom;
        let mut rng = rand::rng();
        online.shuffle(&mut rng);
        online.into_iter().take(config.fanout).collect()
    };

    let msg = GossipMessage::Ping {
        from: state.local_node,
        current_epoch: state.current_epoch(),
        partial_view: state.get_partial_view(10).await,
        failures: state.get_failures().await,
        named_actors: Some(state.get_partial_named_actors(10).await),
    };

    let Ok(payload) = bincode::serialize(&msg) else {
        return;
    };

    for target in targets {
        if let Ok(Some(resp)) = transport.send_gossip(target.addr, payload.clone()).await {
            if let Ok(GossipMessage::Pong {
                from: _,
                current_epoch: _,
                partial_view,
                failures,
                named_actors,
            }) = bincode::deserialize(&resp)
            {
                for node in &partial_view {
                    state.merge_node(node, target.addr).await;
                }
                for f in &failures {
                    state.handle_failure(f).await;
                }
                if let Some(actors) = named_actors {
                    state.merge_named_actors(actors).await;
                }
            }
        }
    }
}

async fn failure_detection_loop(
    state: Arc<ClusterState>,
    config: GossipConfig,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(config.swim.ping_interval);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                detect_failures(&state, &config).await;
            }
            _ = cancel.cancelled() => {
                tracing::info!("Failure detection loop shutting down");
                break;
            }
        }
    }
}

async fn detect_failures(state: &ClusterState, config: &GossipConfig) {
    let now = now_millis();
    let failure_timeout_ms = config.failure_timeout.as_millis() as u64;
    let suspicion_timeout_ms = config.swim.suspicion_timeout.as_millis() as u64;
    let cleanup_ms = config.cleanup_grace_period.as_millis() as u64;

    let mut pfail_nodes = Vec::new();
    let mut fail_nodes = Vec::new();
    let mut remove_nodes = Vec::new();

    {
        let nodes = state.cluster_nodes.read().await;
        for node in nodes.values() {
            if node.node_id == state.local_node {
                continue;
            }

            let elapsed = now.saturating_sub(node.last_seen);

            match node.status {
                NodeStatus::Online if elapsed > failure_timeout_ms => {
                    pfail_nodes.push(node.node_id);
                }
                NodeStatus::PFail if elapsed > suspicion_timeout_ms => {
                    fail_nodes.push(node.node_id);
                }
                NodeStatus::Fail if elapsed > cleanup_ms => {
                    remove_nodes.push(node.node_id);
                }
                _ => {}
            }
        }
    }

    // Mark PFail
    for node_id in pfail_nodes {
        let mut nodes = state.cluster_nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            // Only mark as PFail if still Online (may have been updated by another thread)
            if node.status == NodeStatus::Online {
                node.status = NodeStatus::PFail;
                node.epoch = state.increment_epoch();
                tracing::debug!(
                    node_id = %node_id,
                    elapsed_ms = now.saturating_sub(node.last_seen),
                    "Marked node as PFail"
                );
            }
        }
    }

    // Mark Fail
    for node_id in fail_nodes {
        let mut nodes = state.cluster_nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            // Only mark as Fail if still in PFail state (may have recovered)
            if node.status == NodeStatus::PFail {
                node.status = NodeStatus::Fail;
                node.epoch = state.increment_epoch();
                // Use warn level instead of error to reduce log noise in high-load scenarios
                // This is often a false positive during stress tests
                tracing::warn!(
                    node_id = %node_id,
                    elapsed_ms = now.saturating_sub(node.last_seen),
                    "Marked node as Fail (may be false positive in high-load scenarios)"
                );
            }
        }
    }

    // Cleanup Fail nodes and their named actors
    if !remove_nodes.is_empty() {
        let mut nodes = state.cluster_nodes.write().await;
        let mut named = state.named_actors.write().await;

        for node_id in &remove_nodes {
            if nodes
                .get(node_id)
                .map(|n| n.status == NodeStatus::Fail)
                .unwrap_or(false)
            {
                tracing::info!(node_id = %node_id, "Removing failed node");
                nodes.remove(node_id);

                // Clean up named actors on this node
                for info in named.values_mut() {
                    info.remove_instance(node_id);
                }
            }
        }
        named.retain(|_, info| !info.is_empty());
    }
}

async fn seed_rejoin_loop(
    state: Arc<ClusterState>,
    transport: Arc<Http2Transport>,
    config: GossipConfig,
    interval: Duration,
    cancel: CancellationToken,
) {
    let mut timer = tokio::time::interval(interval);
    timer.tick().await; // Skip first tick

    loop {
        tokio::select! {
            _ = timer.tick() => {
                probe_seeds(&state, &transport, &config).await;
            }
            _ = cancel.cancelled() => {
                tracing::info!("Seed rejoin loop shutting down");
                break;
            }
        }
    }
}

async fn probe_seeds(state: &ClusterState, transport: &Http2Transport, _config: &GossipConfig) {
    let seeds = state.seed_addrs.read().await.clone();
    if seeds.is_empty() {
        return;
    }

    let msg = GossipMessage::Ping {
        from: state.local_node,
        current_epoch: state.current_epoch(),
        partial_view: state.get_partial_view(10).await,
        failures: state.get_failures().await,
        named_actors: Some(state.get_partial_named_actors(10).await),
    };

    let Ok(payload) = bincode::serialize(&msg) else {
        return;
    };

    for addr in &seeds {
        if let Ok(Some(resp)) = transport.send_gossip(*addr, payload.clone()).await {
            if let Ok(GossipMessage::Pong {
                partial_view,
                failures,
                named_actors,
                ..
            }) = bincode::deserialize(&resp)
            {
                for node in &partial_view {
                    state.merge_node(node, *addr).await;
                }
                for f in &failures {
                    state.handle_failure(f).await;
                }
                if let Some(actors) = named_actors {
                    state.merge_named_actors(actors).await;
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gossip_config_default() {
        let config = GossipConfig::default();
        assert_eq!(config.gossip_interval, Duration::from_millis(200));
        assert_eq!(config.fanout, 3);
    }

    #[test]
    fn test_now_millis() {
        let t1 = now_millis();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = now_millis();
        assert!(t2 > t1);
    }

    #[test]
    fn test_fix_addr() {
        let unspecified: SocketAddr = "0.0.0.0:8000".parse().unwrap();
        let peer_ip: std::net::IpAddr = "192.168.1.100".parse().unwrap();
        let fixed = fix_addr(unspecified, peer_ip);
        assert_eq!(fixed.to_string(), "192.168.1.100:8000");

        let specified: SocketAddr = "10.0.0.1:8000".parse().unwrap();
        let not_fixed = fix_addr(specified, peer_ip);
        assert_eq!(not_fixed.to_string(), "10.0.0.1:8000");
    }

    #[test]
    fn test_gossip_message_serialization() {
        let node_id = NodeId::generate();
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        let msg = GossipMessage::Meet {
            from: node_id,
            from_addr: addr,
            current_epoch: 42,
        };

        let payload = bincode::serialize(&msg).unwrap();
        let decoded: GossipMessage = bincode::deserialize(&payload).unwrap();

        match decoded {
            GossipMessage::Meet {
                from,
                from_addr,
                current_epoch,
            } => {
                assert_eq!(from, node_id);
                assert_eq!(from_addr, addr);
                assert_eq!(current_epoch, 42);
            }
            _ => panic!("Expected Meet message"),
        }
    }
}
