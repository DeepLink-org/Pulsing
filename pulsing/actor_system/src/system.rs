//! Actor System - the main entry point for creating and managing actors

use crate::actor::{
    Actor, ActorContext, ActorId, ActorRef, ActorSystemRef, Envelope, Mailbox, NodeId, RawMessage,
};
use crate::cluster::{GossipCluster, GossipConfig, GossipMessage, MemberInfo};
use crate::transport::{HttpMessageHandler, HttpRemoteTransport, HttpTransport, HttpTransportConfig};
use dashmap::DashMap;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Actor runtime statistics
#[derive(Debug, Default)]
pub struct ActorStats {
    /// Number of times the actor started
    pub start_count: AtomicU64,
    /// Number of times the actor stopped
    pub stop_count: AtomicU64,
    /// Number of messages processed
    pub message_count: AtomicU64,
}

impl ActorStats {
    fn inc_start(&self) {
        self.start_count.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_stop(&self) {
        self.stop_count.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_message(&self) {
        self.message_count.fetch_add(1, Ordering::Relaxed);
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "start_count": self.start_count.load(Ordering::Relaxed),
            "stop_count": self.stop_count.load(Ordering::Relaxed),
            "message_count": self.message_count.load(Ordering::Relaxed),
        })
    }
}

/// Actor System configuration
#[derive(Clone, Debug)]
pub struct SystemConfig {
    /// HTTP address for all communication (actors + gossip)
    pub addr: SocketAddr,

    /// Seed nodes to join (HTTP addresses)
    pub seed_nodes: Vec<SocketAddr>,

    /// Gossip configuration
    pub gossip_config: GossipConfig,

    /// HTTP transport configuration
    pub http_config: HttpTransportConfig,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            addr: "0.0.0.0:0".parse().unwrap(),
            seed_nodes: Vec::new(),
            gossip_config: GossipConfig::default(),
            http_config: HttpTransportConfig::default(),
        }
    }
}

impl SystemConfig {
    /// Create config for a standalone node (no cluster)
    pub fn standalone() -> Self {
        Self::default()
    }

    /// Create config with specific address
    pub fn with_addr(addr: SocketAddr) -> Self {
        Self {
            addr,
            ..Default::default()
        }
    }

    /// Add seed nodes for cluster joining
    pub fn with_seeds(mut self, seeds: Vec<SocketAddr>) -> Self {
        self.seed_nodes = seeds;
        self
    }
}

/// Local actor handle
struct LocalActorHandle {
    /// Sender to the actor's mailbox
    sender: mpsc::Sender<Envelope>,

    /// Actor task handle
    join_handle: JoinHandle<()>,

    /// Runtime statistics
    stats: Arc<ActorStats>,

    /// Static metadata provided by the actor
    metadata: HashMap<String, String>,
}

/// The Actor System - manages actors and cluster membership
pub struct ActorSystem {
    /// Local node ID
    node_id: NodeId,

    /// HTTP address
    addr: SocketAddr,

    /// Local actors
    local_actors: Arc<DashMap<String, LocalActorHandle>>,

    /// Gossip cluster (for discovery)
    cluster: Arc<RwLock<Option<Arc<GossipCluster>>>>,

    /// HTTP transport
    transport: Arc<HttpTransport>,

    /// Cancellation token
    cancel_token: CancellationToken,
}

impl ActorSystem {
    /// Create a new actor system
    pub async fn new(config: SystemConfig) -> anyhow::Result<Arc<Self>> {
        let cancel_token = CancellationToken::new();
        let node_id = NodeId::generate();
        let local_actors: Arc<DashMap<String, LocalActorHandle>> = Arc::new(DashMap::new());
        let cluster_holder: Arc<RwLock<Option<Arc<GossipCluster>>>> = Arc::new(RwLock::new(None));

        // Create message handler (needs cluster reference for gossip)
        let handler = SystemMessageHandler {
            node_id: node_id.clone(),
            local_actors: local_actors.clone(),
            cluster: cluster_holder.clone(),
        };

        // Create HTTP transport
        let (transport, actual_addr) = HttpTransport::new(
            config.addr,
            Arc::new(handler),
            config.http_config,
            cancel_token.clone(),
        )
        .await?;

        // Create gossip cluster
        let cluster = GossipCluster::new(
            node_id.clone(),
            actual_addr,
            transport.clone(),
            config.gossip_config,
        );

        let cluster = Arc::new(cluster);
        {
            let mut holder = cluster_holder.write().await;
            *holder = Some(cluster.clone());
        }

        // Start cluster gossip
        cluster.start(cancel_token.clone());

        // Join cluster if seed nodes provided
        if !config.seed_nodes.is_empty() {
            cluster.join(config.seed_nodes).await?;
        }

        let system = Arc::new(Self {
            node_id,
            addr: actual_addr,
            local_actors,
            cluster: cluster_holder,
            transport,
            cancel_token,
        });

        tracing::info!(
            node_id = %system.node_id,
            addr = %system.addr,
            "Actor system started"
        );

        Ok(system)
    }

    /// Get the local node ID
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Get the HTTP address
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Get the HTTP address (alias for compatibility)
    pub fn tcp_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Get the gossip address (same as HTTP address now)
    pub fn gossip_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Spawn a new actor
    pub async fn spawn<A>(self: &Arc<Self>, mut actor: A) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        let actor_id = ActorId::new(self.node_id.clone(), actor.id().name.clone());

        // Create mailbox
        let mailbox = Mailbox::new();
        let sender = mailbox.sender();
        let (_, receiver) = mailbox.split();

        // Create context
        let ctx_system: Arc<dyn ActorSystemRef> = self.clone();
        let mut ctx = ActorContext::with_system(ctx_system, self.cancel_token.clone());
        ctx.set_actor_id(actor_id.clone());

        // Get metadata before starting (as actor is moved)
        let metadata = actor.metadata();
        let stats = Arc::new(ActorStats::default());
        stats.inc_start();

        // Start actor
        if let Err(e) = actor.on_start(&mut ctx).await {
            return Err(anyhow::anyhow!("Actor {} failed to start: {}", actor_id, e));
        }

        // Spawn actor task
        let cancel = self.cancel_token.clone();
        let actor_name = actor_id.name.clone();
        let loop_stats = stats.clone();
        
        let join_handle = tokio::spawn(async move {
            run_actor_loop(actor, receiver, ctx, cancel, loop_stats).await;
        });

        // Register locally
        self.local_actors.insert(
            actor_name.clone(),
            LocalActorHandle {
                sender: sender.clone(),
                join_handle,
                stats,
                metadata,
            },
        );

        // Register in cluster
        {
            let cluster_guard = self.cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                cluster.register_actor(actor_id.clone()).await;
            }
        }

        tracing::debug!(actor_id = %actor_id, "Spawned actor");

        Ok(ActorRef::local(actor_id, sender))
    }

    /// Get a reference to an actor (local or remote)
    pub async fn actor_ref(&self, actor_id: &ActorId) -> anyhow::Result<ActorRef> {
        // Check local first
        if actor_id.node == self.node_id {
            if let Some(handle) = self.local_actors.get(&actor_id.name) {
                return Ok(ActorRef::local(actor_id.clone(), handle.sender.clone()));
            }
        }

        // Check cluster
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            if let Some(member) = cluster.lookup_actor(actor_id).await {
                let transport = Arc::new(HttpRemoteTransport::new(
                    self.transport.clone(),
                    member.addr,
                    actor_id.name.clone(),
                ));
                return Ok(ActorRef::remote(actor_id.clone(), member.addr, transport));
            }
        }

        Err(anyhow::anyhow!("Actor not found: {}", actor_id))
    }

    /// Get cluster members
    pub async fn members(&self) -> Vec<MemberInfo> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            cluster.alive_members().await
        } else {
            Vec::new()
        }
    }

    /// Get local actors
    pub fn local_actor_names(&self) -> Vec<String> {
        self.local_actors.iter().map(|e| e.key().clone()).collect()
    }

    /// Stop an actor
    pub async fn stop(&self, actor_name: &str) -> anyhow::Result<()> {
        if let Some((_, handle)) = self.local_actors.remove(actor_name) {
            handle.join_handle.abort();

            let actor_id = ActorId::new(self.node_id.clone(), actor_name.to_string());
            let cluster_guard = self.cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                cluster.unregister_actor(&actor_id).await;
            }

            tracing::debug!(actor = actor_name, "Stopped actor");
        }
        Ok(())
    }

    /// Shutdown the entire actor system
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        tracing::info!("Shutting down actor system");

        // Signal cancellation
        self.cancel_token.cancel();

        // Leave cluster gracefully
        {
            let cluster_guard = self.cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                cluster.leave().await?;
            }
        }

        // Stop all actors
        for entry in self.local_actors.iter() {
            entry.join_handle.abort();
        }
        self.local_actors.clear();

        Ok(())
    }

    /// Get cancellation token
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }
}

#[async_trait::async_trait]
impl ActorSystemRef for ActorSystem {
    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef> {
        ActorSystem::actor_ref(self, id).await
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

/// Actor message loop
async fn run_actor_loop<A: Actor>(
    mut actor: A,
    mut receiver: mpsc::Receiver<Envelope>,
    mut ctx: ActorContext,
    cancel: CancellationToken,
    stats: Arc<ActorStats>,
) {
    loop {
        tokio::select! {
            Some(envelope) = receiver.recv() => {
                let raw = envelope.to_raw_message();

                stats.inc_message();
                match actor.receive(raw, &mut ctx).await {
                    Ok(response) => {
                        envelope.respond(Ok(response.payload));
                    }
                    Err(e) => {
                        envelope.respond(Err(anyhow::anyhow!("Handler error: {}", e)));
                    }
                }
            }
            _ = cancel.cancelled() => {
                break;
            }
        }
    }

    // Cleanup
    stats.inc_stop();
    if let Err(e) = actor.on_stop(&mut ctx).await {
        tracing::warn!(actor_id = ?ctx.actor_id(), error = %e, "Actor stop error");
    }
}

/// Unified message handler for HTTP transport
struct SystemMessageHandler {
    node_id: NodeId,
    local_actors: Arc<DashMap<String, LocalActorHandle>>,
    cluster: Arc<RwLock<Option<Arc<GossipCluster>>>>,
}

#[async_trait::async_trait]
impl HttpMessageHandler for SystemMessageHandler {
    async fn handle_actor_message(
        &self,
        actor_name: &str,
        msg: RawMessage,
    ) -> anyhow::Result<Vec<u8>> {
        let handle = self
            .local_actors
            .get(actor_name)
            .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", actor_name))?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let envelope = Envelope::ask(msg.msg_type, msg.payload, tx);

        handle
            .sender
            .send(envelope)
            .await
            .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

        rx.await
            .map_err(|_| anyhow::anyhow!("Actor dropped"))?
    }

    async fn handle_gossip_message(&self, payload: Vec<u8>) -> anyhow::Result<Option<Vec<u8>>> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            let msg: GossipMessage = bincode::deserialize(&payload)?;
            let response = cluster.handle_gossip(msg).await?;
            if let Some(resp) = response {
                Ok(Some(bincode::serialize(&resp)?))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    async fn get_node_info(&self) -> serde_json::Value {
        // Collect local actors info
        let mut actors = Vec::new();
        for entry in self.local_actors.iter() {
            let name = entry.key().clone();
            let handle = entry.value();

            actors.push(serde_json::json!({
                "name": name,
                "stats": handle.stats.to_json(),
                "metadata": handle.metadata,
            }));
        }

        // Collect cluster info
        let mut cluster_info = serde_json::json!(null);
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            let members = cluster.alive_members().await;
            
            // Also get remote actors if possible (from gossip cache)
            // Note: GossipCluster doesn't expose all cached actors directly in a convenient way yet,
            // but we can add member count etc.
            cluster_info = serde_json::json!({
                "members_count": members.len(),
                "members": members,
            });
        }

        serde_json::json!({
            "node_id": self.node_id.to_string(),
            "actors": actors,
            "cluster": cluster_info,
        })
    }

    async fn get_actor_info(&self, actor_name: &str) -> Option<serde_json::Value> {
        if let Some(handle) = self.local_actors.get(actor_name) {
            return Some(serde_json::json!({
                "name": actor_name,
                "node_id": self.node_id.to_string(),
                "status": "local",
                "stats": handle.stats.to_json(),
                "metadata": handle.metadata,
            }));
        }

        // Check cluster cache for remote actor location
        // TODO: For remote actors, we would need to either:
        // 1. Scan all known nodes (expensive)
        // 2. Maintain a global actor name index
        // For now, only local actors are queryable by name
        let _cluster_guard = self.cluster.read().await;
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{Message, RawMessage};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug)]
    struct Ping {
        value: i32,
    }

    impl Message for Ping {
        fn type_id() -> &'static str {
            "Ping"
        }
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct Pong {
        result: i32,
    }

    impl Message for Pong {
        fn type_id() -> &'static str {
            "Pong"
        }
    }

    struct CounterActor {
        id: ActorId,
        count: i32,
    }

    #[async_trait::async_trait]
    impl Actor for CounterActor {
        fn id(&self) -> &ActorId {
            &self.id
        }

        async fn receive(
            &mut self,
            msg: RawMessage,
            _ctx: &mut ActorContext,
        ) -> anyhow::Result<RawMessage> {
            match msg.msg_type.as_str() {
                "Ping" => {
                    let ping: Ping = bincode::deserialize(&msg.payload)?;
                    self.count += ping.value;
                    let pong = Pong { result: self.count };
                    RawMessage::from_message(&pong)
                }
                _ => Err(anyhow::anyhow!("Unknown message type: {}", msg.msg_type)),
            }
        }
    }

    #[tokio::test]
    async fn test_actor_system_spawn() {
        let config = SystemConfig::standalone();
        let system = ActorSystem::new(config).await.unwrap();

        let actor = CounterActor {
            id: ActorId::local("counter"),
            count: 0,
        };

        let actor_ref = system.spawn(actor).await.unwrap();
        assert!(actor_ref.is_local());

        // Send message
        let pong: Pong = actor_ref.ask(Ping { value: 10 }).await.unwrap();
        assert_eq!(pong.result, 10);

        let pong: Pong = actor_ref.ask(Ping { value: 5 }).await.unwrap();
        assert_eq!(pong.result, 15);

        system.shutdown().await.unwrap();
    }
}
