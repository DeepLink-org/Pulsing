//! Gossip protocol for cluster membership and actor discovery

use super::member::{ActorLocation, MemberInfo, MemberStatus};
use super::swim::{SwimConfig, SwimDetector, SwimMessage};
use crate::actor::{ActorId, NodeId};
use rand::prelude::IndexedRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Gossip protocol configuration
#[derive(Clone, Debug)]
pub struct GossipConfig {
    /// Interval between gossip rounds
    pub gossip_interval: Duration,

    /// Number of nodes to gossip with per round (fanout)
    pub fanout: usize,

    /// Maximum message size for UDP
    pub max_message_size: usize,

    /// SWIM failure detection config
    pub swim: SwimConfig,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            gossip_interval: Duration::from_millis(200),
            fanout: 3,
            max_message_size: 65507, // Max UDP payload
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
        gossip_addr: SocketAddr,
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

    /// Local TCP address (for actor communication)
    local_addr: SocketAddr,

    /// Local gossip address (UDP)
    gossip_addr: SocketAddr,

    /// Cluster members
    members: Arc<RwLock<HashMap<NodeId, MemberInfo>>>,

    /// Actor registry (actor_id -> node_id)
    actors: Arc<RwLock<HashMap<ActorId, NodeId>>>,

    /// UDP socket for gossip
    socket: Arc<UdpSocket>,

    /// Configuration
    config: GossipConfig,

    /// SWIM failure detector
    #[allow(dead_code)]
    swim: SwimDetector,

    /// Incarnation number (for refuting suspicion)
    #[allow(dead_code)]
    incarnation: Arc<std::sync::atomic::AtomicU64>,
}

impl GossipCluster {
    /// Create a new gossip cluster
    pub async fn new(
        local_addr: SocketAddr,
        gossip_addr: SocketAddr,
        config: GossipConfig,
    ) -> anyhow::Result<Self> {
        let socket = UdpSocket::bind(gossip_addr).await?;
        // Get actual bound address (in case port was 0)
        let actual_gossip_addr = socket.local_addr()?;
        let local_node = NodeId::generate();

        tracing::info!(
            node_id = %local_node,
            tcp_addr = %local_addr,
            gossip_addr = %actual_gossip_addr,
            "Starting gossip cluster"
        );

        Ok(Self {
            local_node: local_node.clone(),
            local_addr,
            gossip_addr: actual_gossip_addr,
            members: Arc::new(RwLock::new(HashMap::new())),
            actors: Arc::new(RwLock::new(HashMap::new())),
            socket: Arc::new(socket),
            swim: SwimDetector::new(local_node, config.swim.clone()),
            config,
            incarnation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Get local node ID
    pub fn local_node(&self) -> &NodeId {
        &self.local_node
    }

    /// Get local TCP address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Get local gossip (UDP) address
    pub fn gossip_addr(&self) -> SocketAddr {
        self.gossip_addr
    }

    /// Join an existing cluster via seed nodes
    pub async fn join(&self, seed_addrs: Vec<SocketAddr>) -> anyhow::Result<()> {
        if seed_addrs.is_empty() {
            tracing::info!("No seed nodes provided, starting as first node");
            return Ok(());
        }

        let msg = GossipMessage::Join {
            node_id: self.local_node.clone(),
            addr: self.local_addr,
            gossip_addr: self.gossip_addr,
        };

        let data = bincode::serialize(&msg)?;

        for addr in seed_addrs {
            tracing::debug!(seed = %addr, "Sending join request");
            self.socket.send_to(&data, addr).await?;
        }

        Ok(())
    }

    /// Start the gossip protocol loops
    pub fn start(&self, cancel_token: CancellationToken) {
        // Receive loop
        let this = self.clone_inner();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            this.receive_loop(cancel).await;
        });

        // Gossip loop
        let this = self.clone_inner();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            this.gossip_loop(cancel).await;
        });

        // SWIM ping loop
        let this = self.clone_inner();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            this.swim_loop(cancel).await;
        });
    }

    /// Clone inner state for spawning tasks
    fn clone_inner(&self) -> GossipClusterInner {
        GossipClusterInner {
            local_node: self.local_node.clone(),
            local_addr: self.local_addr,
            gossip_addr: self.gossip_addr,
            members: self.members.clone(),
            actors: self.actors.clone(),
            socket: self.socket.clone(),
            config: self.config.clone(),
            swim: SwimDetector::new(self.local_node.clone(), self.config.swim.clone()),
            incarnation: self.incarnation.clone(),
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
        let data = bincode::serialize(msg)?;
        let members = self.alive_members().await;

        // Select random targets (before any await)
        let targets: Vec<_> = {
            let mut rng = rand::rng();
            members
                .choose_multiple(&mut rng, self.config.fanout.min(members.len()))
                .cloned()
                .collect()
        };

        for member in targets {
            let _ = self.socket.send_to(&data, member.gossip_addr).await;
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
    gossip_addr: SocketAddr,
    members: Arc<RwLock<HashMap<NodeId, MemberInfo>>>,
    actors: Arc<RwLock<HashMap<ActorId, NodeId>>>,
    socket: Arc<UdpSocket>,
    config: GossipConfig,
    swim: SwimDetector,
    incarnation: Arc<std::sync::atomic::AtomicU64>,
}

impl GossipClusterInner {
    /// Receive loop - handle incoming gossip messages
    async fn receive_loop(&self, cancel: CancellationToken) {
        let mut buf = vec![0u8; self.config.max_message_size];

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, from)) => {
                            if let Ok(msg) = bincode::deserialize::<GossipMessage>(&buf[..len]) {
                                self.handle_message(msg, from).await;
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Gossip receive error");
                        }
                    }
                }
                _ = cancel.cancelled() => {
                    tracing::info!("Gossip receive loop shutting down");
                    break;
                }
            }
        }
    }

    /// Handle incoming gossip message
    async fn handle_message(&self, msg: GossipMessage, from: SocketAddr) {
        match msg {
            GossipMessage::Join {
                node_id,
                addr,
                gossip_addr,
            } => {
                tracing::info!(node_id = %node_id, "Node joining cluster");

                // Add new member
                let member = MemberInfo::new(node_id.clone(), addr, gossip_addr);
                {
                    let mut members = self.members.write().await;
                    members.insert(node_id, member);
                }

                // Send welcome with current state
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

                if let Ok(data) = bincode::serialize(&welcome) {
                    let _ = self.socket.send_to(&data, from).await;
                }
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
            }

            GossipMessage::Sync {
                from: _,
                members,
                actors,
            } => {
                self.merge_members(members).await;
                self.merge_actors(actors).await;
            }

            GossipMessage::Leave { node_id } => {
                tracing::info!(node_id = %node_id, "Node leaving cluster");
                let mut members = self.members.write().await;
                if let Some(member) = members.get_mut(&node_id) {
                    member.status = MemberStatus::Leaving;
                }
            }

            GossipMessage::Swim(swim_msg) => {
                self.handle_swim(swim_msg, from).await;
            }

            GossipMessage::ActorRegistered { location } => {
                let mut actors = self.actors.write().await;
                actors.insert(location.actor_id, location.node_id);
            }

            GossipMessage::ActorUnregistered { actor_id } => {
                let mut actors = self.actors.write().await;
                actors.remove(&actor_id);
            }
        }
    }

    /// Handle SWIM protocol message
    async fn handle_swim(&self, msg: SwimMessage, from: SocketAddr) {
        match msg {
            SwimMessage::Ping { seq, from: _ } => {
                // Send ack
                let ack = self.swim.create_ack(seq);
                let gossip_msg = GossipMessage::Swim(ack);
                if let Ok(data) = bincode::serialize(&gossip_msg) {
                    let _ = self.socket.send_to(&data, from).await;
                }
            }

            SwimMessage::Ack { seq, from: _ } => {
                self.swim.ack_received(seq).await;
            }

            SwimMessage::PingReq {
                seq: _,
                from: _requester,
                target: _,
                target_addr,
            } => {
                // Forward ping to target
                let (_ping_seq, ping) = self.swim.create_ping();
                let gossip_msg = GossipMessage::Swim(ping);
                if let Ok(data) = bincode::serialize(&gossip_msg) {
                    let _ = self.socket.send_to(&data, target_addr).await;
                }

                // TODO: Wait for ack and forward back to requester
            }

            SwimMessage::PingReqAck { seq, from: _, target: _ } => {
                // Mark target as alive
                self.swim.ack_received(seq).await;
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
            // Simple last-write-wins for now
            local.insert(loc.actor_id, loc.node_id);
        }
    }

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

        // Select random targets (do this before any await)
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

        if let Ok(data) = bincode::serialize(&msg) {
            for member in targets {
                let _ = self.socket.send_to(&data, member.gossip_addr).await;
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

        // Pick random target (do this before any await)
        let target = {
            let mut rng = rand::rng();
            members.choose(&mut rng).cloned()
        };

        if let Some(target) = target {
            let (seq, ping) = self.swim.create_ping();
            let msg = GossipMessage::Swim(ping);

            if let Ok(data) = bincode::serialize(&msg) {
                self.swim.ping_sent(seq, target.node_id.clone()).await;
                let _ = self.socket.send_to(&data, target.gossip_addr).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gossip_cluster_creation() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let cluster = GossipCluster::new(addr, addr, GossipConfig::default())
            .await
            .unwrap();

        assert!(!cluster.local_node().0.is_empty());
    }
}

