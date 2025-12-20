//! Gossip protocol for cluster membership and actor discovery
//!
//! Uses HTTP transport for reliable gossip communication.

use super::member::{ActorLocation, MemberInfo, MemberStatus};
use super::swim::{SwimConfig, SwimDetector, SwimMessage};
use crate::actor::{ActorId, NodeId};
use crate::transport::HttpTransport;
use rand::prelude::IndexedRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Gossip protocol configuration
#[derive(Clone, Debug)]
pub struct GossipConfig {
    /// Interval between gossip rounds
    pub gossip_interval: Duration,

    /// Number of nodes to gossip with per round (fanout)
    pub fanout: usize,

    /// Number of times to probe each seed node on startup
    /// Multiple probes through a load-balanced Service IP can discover different pods
    pub seed_probe_count: usize,

    /// Delay between seed probes (to allow load balancer to route to different pods)
    pub seed_probe_interval: Duration,

    /// Interval for periodic seed re-probing after initial join
    /// This helps discover new nodes and recover from network partitions
    /// Set to None to disable periodic re-probing
    pub seed_rejoin_interval: Option<Duration>,

    /// SWIM failure detection config
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
            swim: SwimConfig::default(),
        }
    }
}

/// Gossip protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipMessage {
    /// Join request from a new node
    Join {
        node_id: NodeId,
        addr: SocketAddr,
    },

    /// Welcome response with cluster state
    Welcome {
        from: NodeId,
        members: Vec<MemberInfo>,
        actors: Vec<ActorLocation>,
    },

    /// Periodic sync (piggyback on heartbeat)
    Sync {
        from: NodeId,
        members: Vec<MemberInfo>,
        actors: Vec<ActorLocation>,
    },

    /// Node is leaving gracefully
    Leave { node_id: NodeId },

    /// SWIM failure detection message
    Swim(SwimMessage),

    /// Actor registered
    ActorRegistered { location: ActorLocation },

    /// Actor unregistered
    ActorUnregistered { actor_id: ActorId },
}

/// Gossip cluster state
pub struct GossipCluster {
    /// Local node ID
    local_node: NodeId,

    /// Local HTTP address
    local_addr: SocketAddr,

    /// Cluster members
    members: Arc<RwLock<HashMap<NodeId, MemberInfo>>>,

    /// Actor registry (actor_id -> node_id)
    actors: Arc<RwLock<HashMap<ActorId, NodeId>>>,

    /// HTTP transport for gossip
    transport: Arc<HttpTransport>,

    /// Seed addresses for periodic re-probing
    seed_addrs: Arc<RwLock<Vec<SocketAddr>>>,

    /// Configuration
    config: GossipConfig,

    /// SWIM failure detector
    swim: SwimDetector,

    /// Incarnation number (for refuting suspicion)
    #[allow(dead_code)]
    incarnation: Arc<std::sync::atomic::AtomicU64>,
}

impl GossipCluster {
    /// Create a new gossip cluster
    pub fn new(
        local_node: NodeId,
        local_addr: SocketAddr,
        transport: Arc<HttpTransport>,
        config: GossipConfig,
    ) -> Self {
        tracing::info!(
            node_id = %local_node,
            addr = %local_addr,
            "Starting gossip cluster"
        );

        Self {
            local_node: local_node.clone(),
            local_addr,
            members: Arc::new(RwLock::new(HashMap::new())),
            actors: Arc::new(RwLock::new(HashMap::new())),
            transport,
            seed_addrs: Arc::new(RwLock::new(Vec::new())),
            swim: SwimDetector::new(local_node, config.swim.clone()),
            config,
            incarnation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Get local node ID
    pub fn local_node(&self) -> &NodeId {
        &self.local_node
    }

    /// Get local address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Join an existing cluster via seed nodes
    /// 
    /// Probes each seed address multiple times to discover more peers through
    /// load-balanced endpoints (e.g., Kubernetes Service IP).
    pub async fn join(&self, seed_addrs: Vec<SocketAddr>) -> anyhow::Result<()> {
        if seed_addrs.is_empty() {
            tracing::info!("No seed nodes provided, starting as first node");
            return Ok(());
        }

        // Store seed addresses for periodic re-probing
        {
            let mut stored_seeds = self.seed_addrs.write().await;
            *stored_seeds = seed_addrs.clone();
        }

        let msg = GossipMessage::Join {
            node_id: self.local_node.clone(),
            addr: self.local_addr,
        };

        let payload = bincode::serialize(&msg)?;
        let probe_count = self.config.seed_probe_count.max(1);
        let probe_interval = self.config.seed_probe_interval;

        tracing::info!(
            seeds = ?seed_addrs,
            probe_count = probe_count,
            "Probing seed nodes to discover cluster"
        );

        for probe in 0..probe_count {
            for addr in &seed_addrs {
                tracing::debug!(seed = %addr, probe = probe + 1, "Sending join request");
                match self.transport.send_gossip(*addr, payload.clone()).await {
                    Ok(Some(response_payload)) => {
                        if let Ok(response) = bincode::deserialize::<GossipMessage>(&response_payload) {
                            self.handle_gossip(response).await?;
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        tracing::warn!(seed = %addr, error = %e, "Failed to join seed node");
                    }
                }
            }

            // Check if we've discovered enough peers
            let member_count = self.members.read().await.len();
            if member_count >= self.config.fanout {
                tracing::info!(
                    members = member_count,
                    probes = probe + 1,
                    "Discovered enough peers, stopping seed probing"
                );
                break;
            }

            // Wait before next probe round (allows LB to route to different pods)
            if probe < probe_count - 1 {
                tokio::time::sleep(probe_interval).await;
            }
        }

        let final_count = self.members.read().await.len();
        tracing::info!(members = final_count, "Seed probing complete");

        Ok(())
    }

    /// Start the gossip protocol loops
    pub fn start(&self, cancel_token: CancellationToken) {
        // Gossip loop
        let cluster = self.clone_inner();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            cluster.gossip_loop(cancel).await;
        });

        // SWIM ping loop
        let cluster = self.clone_inner();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            cluster.swim_loop(cancel).await;
        });

        // Periodic seed re-probe loop (if enabled)
        if let Some(interval) = self.config.seed_rejoin_interval {
            let cluster = self.clone_inner();
            let cancel = cancel_token.clone();
            tokio::spawn(async move {
                cluster.seed_rejoin_loop(interval, cancel).await;
            });
        }
    }

    /// Clone inner state for spawning tasks
    fn clone_inner(&self) -> GossipClusterInner {
        GossipClusterInner {
            local_node: self.local_node.clone(),
            local_addr: self.local_addr,
            members: self.members.clone(),
            actors: self.actors.clone(),
            transport: self.transport.clone(),
            seed_addrs: self.seed_addrs.clone(),
            config: self.config.clone(),
            swim: self.swim.clone(),
            incarnation: self.incarnation.clone(),
        }
    }

    /// Handle incoming gossip message (called by HTTP handler)
    pub async fn handle_gossip(&self, msg: GossipMessage) -> anyhow::Result<Option<GossipMessage>> {
        match msg {
            GossipMessage::Join { node_id, addr } => {
                tracing::info!(node_id = %node_id, "Node joining cluster");

                // Add new member (same addr for both since we use single port)
                let member = MemberInfo::new(node_id.clone(), addr, addr);
                {
                    let mut members = self.members.write().await;
                    members.insert(node_id, member);
                }

                // Return welcome with current state
                let welcome = GossipMessage::Welcome {
                    from: self.local_node.clone(),
                    members: self.members.read().await.values().cloned().collect(),
                    actors: self
                        .actors
                        .read()
                        .await
                        .iter()
                        .map(|(id, node)| ActorLocation::new(id.clone(), node.clone()))
                        .collect(),
                };

                Ok(Some(welcome))
            }

            GossipMessage::Welcome {
                from: _,
                members,
                actors,
            } => {
                tracing::debug!(
                    member_count = members.len(),
                    actor_count = actors.len(),
                    "Received welcome"
                );
                self.merge_members(members).await;
                self.merge_actors(actors).await;
                Ok(None)
            }

            GossipMessage::Sync {
                from: _,
                members,
                actors,
            } => {
                self.merge_members(members).await;
                self.merge_actors(actors).await;
                Ok(None)
            }

            GossipMessage::Leave { node_id } => {
                tracing::info!(node_id = %node_id, "Node leaving cluster");
                let mut members = self.members.write().await;
                if let Some(member) = members.get_mut(&node_id) {
                    member.status = MemberStatus::Leaving;
                }
                Ok(None)
            }

            GossipMessage::Swim(swim_msg) => {
                self.handle_swim(swim_msg).await
            }

            GossipMessage::ActorRegistered { location } => {
                let mut actors = self.actors.write().await;
                actors.insert(location.actor_id, location.node_id);
                Ok(None)
            }

            GossipMessage::ActorUnregistered { actor_id } => {
                let mut actors = self.actors.write().await;
                actors.remove(&actor_id);
                Ok(None)
            }
        }
    }

    /// Handle SWIM protocol message
    async fn handle_swim(&self, msg: SwimMessage) -> anyhow::Result<Option<GossipMessage>> {
        match msg {
            SwimMessage::Ping { seq, from: _ } => {
                let ack = self.swim.create_ack(seq);
                Ok(Some(GossipMessage::Swim(ack)))
            }

            SwimMessage::Ack { seq, from: _ } => {
                self.swim.ack_received(seq).await;
                Ok(None)
            }

            SwimMessage::PingReq { .. } => {
                // TODO: Implement indirect ping
                Ok(None)
            }

            SwimMessage::PingReqAck { seq, from: _, target: _ } => {
                self.swim.ack_received(seq).await;
                Ok(None)
            }
        }
    }

    /// Merge received member list with local state
    async fn merge_members(&self, remote_members: Vec<MemberInfo>) {
        let mut local = self.members.write().await;

        for remote in remote_members {
            match local.get(&remote.node_id) {
                Some(existing) if existing.supersedes(&remote) => {
                    // Local version is newer, ignore
                }
                _ => {
                    local.insert(remote.node_id.clone(), remote);
                }
            }
        }
    }

    /// Merge received actor locations with local state
    async fn merge_actors(&self, remote_actors: Vec<ActorLocation>) {
        let mut local = self.actors.write().await;

        for loc in remote_actors {
            local.insert(loc.actor_id, loc.node_id);
        }
    }

    /// Register a local actor
    pub async fn register_actor(&self, actor_id: ActorId) {
        let mut actors = self.actors.write().await;
        actors.insert(actor_id.clone(), self.local_node.clone());

        // Broadcast to cluster
        let location = ActorLocation::new(actor_id, self.local_node.clone());
        let msg = GossipMessage::ActorRegistered { location };
        let _ = self.broadcast_message(&msg).await;
    }

    /// Unregister a local actor
    pub async fn unregister_actor(&self, actor_id: &ActorId) {
        let mut actors = self.actors.write().await;
        actors.remove(actor_id);

        // Broadcast to cluster
        let msg = GossipMessage::ActorUnregistered {
            actor_id: actor_id.clone(),
        };
        let _ = self.broadcast_message(&msg).await;
    }

    /// Lookup an actor's location
    pub async fn lookup_actor(&self, actor_id: &ActorId) -> Option<MemberInfo> {
        let actors = self.actors.read().await;
        let node_id = actors.get(actor_id)?;

        let members = self.members.read().await;
        members.get(node_id).cloned()
    }

    /// Get all alive members
    pub async fn alive_members(&self) -> Vec<MemberInfo> {
        self.members
            .read()
            .await
            .values()
            .filter(|m| m.status.is_alive())
            .cloned()
            .collect()
    }

    /// Get member by node ID
    pub async fn get_member(&self, node_id: &NodeId) -> Option<MemberInfo> {
        self.members.read().await.get(node_id).cloned()
    }

    /// Broadcast a message to random members
    async fn broadcast_message(&self, msg: &GossipMessage) -> anyhow::Result<()> {
        let payload = bincode::serialize(msg)?;
        let members = self.alive_members().await;

        // Select random targets
        let targets: Vec<_> = {
            let mut rng = rand::rng();
            members
                .choose_multiple(&mut rng, self.config.fanout.min(members.len()))
                .cloned()
                .collect()
        };

        for member in targets {
            let _ = self.transport.send_gossip(member.addr, payload.clone()).await;
        }

        Ok(())
    }

    /// Leave the cluster gracefully
    pub async fn leave(&self) -> anyhow::Result<()> {
        let msg = GossipMessage::Leave {
            node_id: self.local_node.clone(),
        };
        self.broadcast_message(&msg).await
    }
}

/// Inner state for async tasks
#[allow(dead_code)]
struct GossipClusterInner {
    local_node: NodeId,
    local_addr: SocketAddr,
    members: Arc<RwLock<HashMap<NodeId, MemberInfo>>>,
    actors: Arc<RwLock<HashMap<ActorId, NodeId>>>,
    transport: Arc<HttpTransport>,
    seed_addrs: Arc<RwLock<Vec<SocketAddr>>>,
    config: GossipConfig,
    swim: SwimDetector,
    #[allow(dead_code)]
    incarnation: Arc<std::sync::atomic::AtomicU64>,
}

impl GossipClusterInner {
    /// Gossip loop - periodically sync with random members
    async fn gossip_loop(&self, cancel: CancellationToken) {
        let mut interval = tokio::time::interval(self.config.gossip_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    self.gossip_round().await;
                }
                _ = cancel.cancelled() => {
                    tracing::info!("Gossip loop shutting down");
                    break;
                }
            }
        }
    }

    /// One round of gossip
    async fn gossip_round(&self) {
        let members: Vec<_> = self
            .members
            .read()
            .await
            .values()
            .filter(|m| m.status.is_reachable() && m.node_id != self.local_node)
            .cloned()
            .collect();

        if members.is_empty() {
            return;
        }

        // Select random targets
        let targets: Vec<_> = {
            let mut rng = rand::rng();
            members
                .choose_multiple(&mut rng, self.config.fanout.min(members.len()))
                .cloned()
                .collect()
        };

        // Build sync message
        let msg = GossipMessage::Sync {
            from: self.local_node.clone(),
            members: self.members.read().await.values().cloned().collect(),
            actors: self
                .actors
                .read()
                .await
                .iter()
                .map(|(id, node)| ActorLocation::new(id.clone(), node.clone()))
                .collect(),
        };

        if let Ok(payload) = bincode::serialize(&msg) {
            for member in targets {
                let _ = self.transport.send_gossip(member.addr, payload.clone()).await;
            }
        }
    }

    /// SWIM failure detection loop
    async fn swim_loop(&self, cancel: CancellationToken) {
        let mut interval = tokio::time::interval(self.swim.ping_interval());

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Ping a random member
                    self.swim_ping_round().await;

                    // Check for timeouts
                    let timeouts = self.swim.check_timeouts().await;
                    for (node_id, should_suspect) in timeouts {
                        if should_suspect {
                            let mut members = self.members.write().await;
                            if let Some(member) = members.get_mut(&node_id) {
                                member.suspect();
                                tracing::warn!(node_id = %node_id, "Suspecting node");
                            }
                        }
                    }
                }
                _ = cancel.cancelled() => {
                    tracing::info!("SWIM loop shutting down");
                    break;
                }
            }
        }
    }

    /// Periodic seed re-probe loop
    /// 
    /// Periodically probes seed addresses to:
    /// - Discover new nodes that joined via the same Service IP
    /// - Recover from network partitions
    /// - Maintain cluster connectivity
    async fn seed_rejoin_loop(&self, interval: Duration, cancel: CancellationToken) {
        let mut timer = tokio::time::interval(interval);
        
        // Skip the first tick (we already probed on startup)
        timer.tick().await;

        loop {
            tokio::select! {
                _ = timer.tick() => {
                    self.probe_seeds().await;
                }
                _ = cancel.cancelled() => {
                    tracing::info!("Seed rejoin loop shutting down");
                    break;
                }
            }
        }
    }

    /// Probe all seed addresses once
    async fn probe_seeds(&self) {
        let seeds = self.seed_addrs.read().await.clone();
        if seeds.is_empty() {
            return;
        }

        let msg = GossipMessage::Sync {
            from: self.local_node.clone(),
            members: self.members.read().await.values().cloned().collect(),
            actors: self
                .actors
                .read()
                .await
                .iter()
                .map(|(id, node)| ActorLocation::new(id.clone(), node.clone()))
                .collect(),
        };

        let payload = match bincode::serialize(&msg) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to serialize sync message for seed probe");
                return;
            }
        };

        let before_count = self.members.read().await.len();

        for addr in &seeds {
            match self.transport.send_gossip(*addr, payload.clone()).await {
                Ok(Some(response_payload)) => {
                    if let Ok(response) = bincode::deserialize::<GossipMessage>(&response_payload) {
                        // Merge received state
                        if let GossipMessage::Sync { members, actors, .. } 
                            | GossipMessage::Welcome { members, actors, .. } = response 
                        {
                            self.merge_members(members).await;
                            self.merge_actors(actors).await;
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::debug!(seed = %addr, error = %e, "Seed probe failed");
                }
            }
        }

        let after_count = self.members.read().await.len();
        if after_count > before_count {
            tracing::info!(
                before = before_count,
                after = after_count,
                "Discovered new members via seed probe"
            );
        }
    }

    /// Ping a random alive member
    async fn swim_ping_round(&self) {
        let members: Vec<_> = self
            .members
            .read()
            .await
            .values()
            .filter(|m| m.status.is_alive() && m.node_id != self.local_node)
            .cloned()
            .collect();

        if members.is_empty() {
            return;
        }

        // Pick random target
        let target = {
            let mut rng = rand::rng();
            members.choose(&mut rng).cloned()
        };

        if let Some(target) = target {
            let (seq, ping) = self.swim.create_ping();
            let msg = GossipMessage::Swim(ping);

            if let Ok(payload) = bincode::serialize(&msg) {
                self.swim.ping_sent(seq, target.node_id.clone()).await;
                let _ = self.transport.send_gossip(target.addr, payload).await;
            }
        }
    }

    /// Merge received member list with local state
    async fn merge_members(&self, remote_members: Vec<MemberInfo>) {
        let mut local = self.members.write().await;

        for remote in remote_members {
            match local.get(&remote.node_id) {
                Some(existing) if existing.supersedes(&remote) => {
                    // Local version is newer, ignore
                }
                _ => {
                    local.insert(remote.node_id.clone(), remote);
                }
            }
        }
    }

    /// Merge received actor locations with local state
    async fn merge_actors(&self, remote_actors: Vec<ActorLocation>) {
        let mut local = self.actors.write().await;

        for loc in remote_actors {
            local.insert(loc.actor_id, loc.node_id);
        }
    }
}

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
    fn test_gossip_message_serialization() {
        let msg = GossipMessage::Join {
            node_id: NodeId::generate(),
            addr: "127.0.0.1:8000".parse().unwrap(),
        };

        let payload = bincode::serialize(&msg).unwrap();
        let decoded: GossipMessage = bincode::deserialize(&payload).unwrap();

        match decoded {
            GossipMessage::Join { node_id: _, addr } => {
                assert_eq!(addr, "127.0.0.1:8000".parse::<SocketAddr>().unwrap());
            }
            _ => panic!("Expected Join message"),
        }
    }
}
