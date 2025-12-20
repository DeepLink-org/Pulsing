//! Actor System - the main entry point for creating and managing actors

use crate::actor::{
    Actor, ActorContext, ActorId, ActorRef, ActorSystemRef, Envelope, Mailbox, MessageHandler,
    NodeId,
};
use crate::cluster::{GossipCluster, GossipConfig, MemberInfo};
use crate::transport::{TcpRemoteTransport, TcpTransport, TcpTransportConfig};
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Actor System configuration
#[derive(Clone, Debug)]
pub struct SystemConfig {
    /// TCP address for actor communication
    pub tcp_addr: SocketAddr,

    /// Gossip address for cluster membership
    pub gossip_addr: SocketAddr,

    /// Seed nodes to join
    pub seed_nodes: Vec<SocketAddr>,

    /// Gossip configuration
    pub gossip_config: GossipConfig,

    /// TCP transport configuration
    pub tcp_config: TcpTransportConfig,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            tcp_addr: "0.0.0.0:0".parse().unwrap(),
            gossip_addr: "0.0.0.0:0".parse().unwrap(),
            seed_nodes: Vec::new(),
            gossip_config: GossipConfig::default(),
            tcp_config: TcpTransportConfig::default(),
        }
    }
}

impl SystemConfig {
    /// Create config for a standalone node (no cluster)
    pub fn standalone() -> Self {
        Self::default()
    }

    /// Create config with specific addresses
    pub fn with_addrs(tcp_addr: SocketAddr, gossip_addr: SocketAddr) -> Self {
        Self {
            tcp_addr,
            gossip_addr,
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
}

/// The Actor System - manages actors and cluster membership
pub struct ActorSystem {
    /// Local node ID
    node_id: NodeId,

    /// Local actors
    local_actors: Arc<DashMap<String, LocalActorHandle>>,

    /// Gossip cluster (for discovery)
    cluster: Arc<GossipCluster>,

    /// TCP transport (for actor communication)
    transport: Arc<TcpTransport>,

    /// Cancellation token
    cancel_token: CancellationToken,
}

impl ActorSystem {
    /// Create a new actor system
    pub async fn new(config: SystemConfig) -> anyhow::Result<Arc<Self>> {
        let cancel_token = CancellationToken::new();

        // Create cluster first to get node ID
        let cluster = GossipCluster::new(
            config.tcp_addr,
            config.gossip_addr,
            config.gossip_config,
        )
        .await?;

        let node_id = cluster.local_node().clone();
        let local_actors: Arc<DashMap<String, LocalActorHandle>> = Arc::new(DashMap::new());

        // Create message handler
        let handler = ActorMessageHandler {
            local_actors: local_actors.clone(),
        };

        // Create TCP transport
        let transport = TcpTransport::new(
            config.tcp_addr,
            Arc::new(handler),
            config.tcp_config,
        )
        .await?;

        let cluster = Arc::new(cluster);

        // Start cluster gossip
        cluster.start(cancel_token.clone());

        // Join cluster if seed nodes provided
        if !config.seed_nodes.is_empty() {
            cluster.join(config.seed_nodes).await?;
        }

        let system = Arc::new(Self {
            node_id,
            local_actors,
            cluster,
            transport,
            cancel_token,
        });

        tracing::info!(
            node_id = %system.node_id,
            tcp_addr = %system.transport.local_addr(),
            "Actor system started"
        );

        Ok(system)
    }

    /// Get the local node ID
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Get the TCP address
    pub fn tcp_addr(&self) -> SocketAddr {
        self.transport.local_addr()
    }

    /// Get the gossip (UDP) address
    pub fn gossip_addr(&self) -> SocketAddr {
        self.cluster.gossip_addr()
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

        // Start actor
        if let Err(e) = actor.on_start(&mut ctx).await {
            return Err(anyhow::anyhow!("Actor {} failed to start: {}", actor_id, e));
        }

        // Spawn actor task
        let cancel = self.cancel_token.clone();
        let actor_name = actor_id.name.clone();
        let join_handle = tokio::spawn(async move {
            run_actor_loop(actor, receiver, ctx, cancel).await;
        });

        // Register locally
        self.local_actors.insert(
            actor_name.clone(),
            LocalActorHandle {
                sender: sender.clone(),
                join_handle,
            },
        );

        // Register in cluster
        self.cluster.register_actor(actor_id.clone()).await;

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
        if let Some(member) = self.cluster.lookup_actor(actor_id).await {
            let transport = Arc::new(TcpRemoteTransport::new(
                self.transport.clone(),
                member.addr,
            ));
            return Ok(ActorRef::remote(actor_id.clone(), member.addr, transport));
        }

        Err(anyhow::anyhow!("Actor not found: {}", actor_id))
    }

    /// Get cluster members
    pub async fn members(&self) -> Vec<MemberInfo> {
        self.cluster.alive_members().await
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
            self.cluster.unregister_actor(&actor_id).await;

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
        self.cluster.leave().await?;

        // Stop all actors
        for entry in self.local_actors.iter() {
            entry.join_handle.abort();
        }
        self.local_actors.clear();

        // Shutdown transport
        self.transport.shutdown();

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
) {
    loop {
        tokio::select! {
            Some(envelope) = receiver.recv() => {
                let raw = envelope.to_raw_message();

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
    if let Err(e) = actor.on_stop(&mut ctx).await {
        tracing::warn!(actor_id = ?ctx.actor_id(), error = %e, "Actor stop error");
    }
}

/// Message handler for incoming TCP requests
struct ActorMessageHandler {
    local_actors: Arc<DashMap<String, LocalActorHandle>>,
}

#[async_trait::async_trait]
impl MessageHandler for ActorMessageHandler {
    async fn handle_message(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        let handle = self
            .local_actors
            .get(&actor_id.name)
            .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", actor_id))?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let envelope = Envelope::ask(msg_type.to_string(), payload, tx);

        handle
            .sender
            .send(envelope)
            .await
            .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

        rx.await
            .map_err(|_| anyhow::anyhow!("Actor dropped"))?
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

