//! Actor System - the main entry point for creating and managing actors

use crate::actor::{
    Actor, ActorAddress, ActorContext, ActorId, ActorPath, ActorRef, ActorSystemRef, Envelope,
    Mailbox, NodeId, RawMessage, StopReason, LOCALHOST,
};
use crate::cluster::{GossipCluster, GossipConfig, GossipMessage, MemberInfo, NamedActorInfo};
use crate::transport::{
    HttpMessageHandler, HttpRemoteTransport, HttpTransport, HttpTransportConfig,
};
use crate::watch::ActorLifecycle;
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

    /// Named actor path (if this is a named actor)
    named_path: Option<ActorPath>,

    /// Full actor ID
    actor_id: ActorId,
}

/// The Actor System - manages actors and cluster membership
pub struct ActorSystem {
    /// Local node ID
    node_id: NodeId,

    /// HTTP address
    addr: SocketAddr,

    /// Local actors (actor_name -> handle)
    local_actors: Arc<DashMap<String, LocalActorHandle>>,

    /// Named actor path to local actor name mapping (path_string -> actor_name)
    named_actor_paths: Arc<DashMap<String, String>>,

    /// Gossip cluster (for discovery)
    cluster: Arc<RwLock<Option<Arc<GossipCluster>>>>,

    /// HTTP transport
    transport: Arc<HttpTransport>,

    /// Cancellation token
    cancel_token: CancellationToken,

    /// Actor lifecycle manager (watch, termination handling)
    lifecycle: Arc<ActorLifecycle>,
}

impl ActorSystem {
    /// Create a new actor system
    pub async fn new(config: SystemConfig) -> anyhow::Result<Arc<Self>> {
        let cancel_token = CancellationToken::new();
        let node_id = NodeId::generate();
        let local_actors: Arc<DashMap<String, LocalActorHandle>> = Arc::new(DashMap::new());
        let named_actor_paths: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
        let cluster_holder: Arc<RwLock<Option<Arc<GossipCluster>>>> = Arc::new(RwLock::new(None));
        let lifecycle = Arc::new(ActorLifecycle::new());

        // Create message handler (needs cluster reference for gossip)
        let handler = SystemMessageHandler {
            node_id: node_id.clone(),
            local_actors: local_actors.clone(),
            named_actor_paths: named_actor_paths.clone(),
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
            named_actor_paths,
            cluster: cluster_holder,
            transport,
            cancel_token,
            lifecycle,
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

    /// Spawn a new actor (not registered in cluster)
    ///
    /// This creates a local actor that is not broadcast to the cluster.
    /// Use `spawn_named` for actors that need to be discoverable cluster-wide.
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

        // Create termination notification channel
        let (term_tx, term_rx) = tokio::sync::oneshot::channel::<StopReason>();

        // Spawn actor task
        let cancel = self.cancel_token.clone();
        let actor_name = actor_id.name.clone();
        let loop_stats = stats.clone();

        let join_handle = tokio::spawn(async move {
            let reason = run_actor_loop(actor, receiver, ctx, cancel, loop_stats).await;
            let _ = term_tx.send(reason);
        });

        // Register locally
        self.local_actors.insert(
            actor_name.clone(),
            LocalActorHandle {
                sender: sender.clone(),
                join_handle,
                stats,
                metadata,
                named_path: None,
                actor_id: actor_id.clone(),
            },
        );

        // Spawn termination handler task
        let system = self.clone();
        let term_actor_id = actor_id.clone();
        let term_actor_name = actor_name.clone();
        tokio::spawn(async move {
            if let Ok(reason) = term_rx.await {
                system
                    .handle_actor_terminated(&term_actor_name, &term_actor_id, None, reason)
                    .await;
            }
        });

        tracing::debug!(actor_id = %actor_id, "Spawned actor");

        Ok(ActorRef::local(actor_id, sender))
    }

    /// Spawn a named actor with a path (broadcasts location to cluster)
    ///
    /// Named actors are registered in the cluster registry and can be discovered
    /// by other nodes using the actor path. Multiple instances can be deployed
    /// on different nodes.
    ///
    /// # Arguments
    /// * `path` - The actor path (e.g., "services/llm/router")
    /// * `actor` - The actor implementation
    ///
    /// # Example
    /// ```ignore
    /// let path = ActorPath::new("services/llm/router")?;
    /// let actor_ref = system.spawn_named(path, MyActor::new()).await?;
    /// ```
    pub async fn spawn_named<A>(
        self: &Arc<Self>,
        path: ActorPath,
        mut actor: A,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        let path_key = path.as_str();

        // Check if path is already registered locally
        if self.named_actor_paths.contains_key(&path_key) {
            return Err(anyhow::anyhow!(
                "Named actor path '{}' is already registered on this node",
                path_key
            ));
        }

        let actor_id = ActorId::new(self.node_id.clone(), actor.id().name.clone());

        // Create mailbox
        let mailbox = Mailbox::new();
        let sender = mailbox.sender();
        let (_, receiver) = mailbox.split();

        // Create context
        let ctx_system: Arc<dyn ActorSystemRef> = self.clone();
        let mut ctx = ActorContext::with_system(ctx_system, self.cancel_token.clone());
        ctx.set_actor_id(actor_id.clone());

        // Get metadata before starting
        let metadata = actor.metadata();
        let stats = Arc::new(ActorStats::default());
        stats.inc_start();

        // Start actor
        if let Err(e) = actor.on_start(&mut ctx).await {
            return Err(anyhow::anyhow!("Actor {} failed to start: {}", actor_id, e));
        }

        // Create termination notification channel
        let (term_tx, term_rx) = tokio::sync::oneshot::channel::<StopReason>();

        // Spawn actor task
        let cancel = self.cancel_token.clone();
        let actor_name = actor_id.name.clone();
        let loop_stats = stats.clone();

        let join_handle = tokio::spawn(async move {
            let reason = run_actor_loop(actor, receiver, ctx, cancel, loop_stats).await;
            let _ = term_tx.send(reason);
        });

        // Register locally with path
        self.local_actors.insert(
            actor_name.clone(),
            LocalActorHandle {
                sender: sender.clone(),
                join_handle,
                stats,
                metadata,
                named_path: Some(path.clone()),
                actor_id: actor_id.clone(),
            },
        );

        // Register path -> actor_name mapping
        self.named_actor_paths
            .insert(path_key.clone(), actor_name.clone());

        // Register in cluster (broadcast)
        {
            let cluster_guard = self.cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                cluster.register_named_actor(path.clone()).await;
            }
        }

        // Spawn termination handler task
        let system = self.clone();
        let term_actor_id = actor_id.clone();
        let term_actor_name = actor_name.clone();
        let term_path = path.clone();
        tokio::spawn(async move {
            if let Ok(reason) = term_rx.await {
                system
                    .handle_actor_terminated(
                        &term_actor_name,
                        &term_actor_id,
                        Some(term_path),
                        reason,
                    )
                    .await;
            }
        });

        tracing::debug!(
            actor_id = %actor_id,
            path = %path,
            "Spawned named actor"
        );

        Ok(ActorRef::local(actor_id, sender))
    }

    /// Get a reference to an actor by ActorId (local or remote)
    ///
    /// For the new addressing scheme, prefer using `resolve()` with `ActorAddress`.
    pub async fn actor_ref(&self, actor_id: &ActorId) -> anyhow::Result<ActorRef> {
        // Check local first
        if actor_id.node == self.node_id {
            if let Some(handle) = self.local_actors.get(&actor_id.name) {
                return Ok(ActorRef::local(actor_id.clone(), handle.sender.clone()));
            }
            return Err(anyhow::anyhow!("Local actor not found: {}", actor_id));
        }

        // Check cluster
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            // 1. Direct addressing: check if we know the node
            if let Some(member) = cluster.get_member(&actor_id.node).await {
                let transport = Arc::new(HttpRemoteTransport::new(
                    self.transport.clone(),
                    member.addr,
                    actor_id.name.clone(),
                ));
                return Ok(ActorRef::remote(actor_id.clone(), member.addr, transport));
            }

            // 2. Fallback: check gossip cache (for legacy or alias support)
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

    /// Resolve an actor address to get an ActorRef
    ///
    /// This is the primary method for getting actor references using the new
    /// addressing scheme.
    ///
    /// # Address Types
    /// - Named service: `actor:///namespace/path/name` - load-balanced access
    /// - Named instance: `actor:///namespace/path/name@node_id` - specific instance
    /// - Global: `actor://node_id/actor_id` - direct address
    /// - Local: `actor://localhost/actor_id` - current node shortcut
    ///
    /// # Example
    /// ```ignore
    /// // Access named actor (load balanced)
    /// let addr = ActorAddress::parse("actor:///services/llm/router")?;
    /// let actor_ref = system.resolve(&addr).await?;
    ///
    /// // Access specific instance
    /// let addr = ActorAddress::parse("actor:///services/llm/router@node_a")?;
    /// let actor_ref = system.resolve(&addr).await?;
    ///
    /// // Access by global address
    /// let addr = ActorAddress::parse("actor://node_a/worker_123")?;
    /// let actor_ref = system.resolve(&addr).await?;
    /// ```
    pub async fn resolve(&self, address: &ActorAddress) -> anyhow::Result<ActorRef> {
        match address {
            ActorAddress::Named { path, instance } => {
                self.resolve_named(path, instance.as_ref()).await
            }
            ActorAddress::Global { node_id, actor_id } => {
                self.resolve_global(node_id, actor_id).await
            }
        }
    }

    /// Resolve a named actor address
    async fn resolve_named(
        &self,
        path: &ActorPath,
        instance: Option<&NodeId>,
    ) -> anyhow::Result<ActorRef> {
        let path_key = path.as_str();

        // If instance is specified, resolve to that specific node
        if let Some(target_node) = instance {
            // Check if it's local
            if target_node == &self.node_id {
                if let Some(actor_name) = self.named_actor_paths.get(&path_key) {
                    if let Some(handle) = self.local_actors.get(actor_name.value()) {
                        let actor_id = ActorId::new(self.node_id.clone(), actor_name.clone());
                        return Ok(ActorRef::local(actor_id, handle.sender.clone()));
                    }
                }
                return Err(anyhow::anyhow!(
                    "Named actor not found locally: {}",
                    path_key
                ));
            }

            // Remote instance - need to find the node and route to it
            let cluster_guard = self.cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                if let Some(member) = cluster.get_member(target_node).await {
                    // Use path name as the actor name for remote routing
                    let transport = Arc::new(HttpRemoteTransport::new_named(
                        self.transport.clone(),
                        member.addr,
                        path.clone(),
                    ));
                    let actor_id = ActorId::new(target_node.clone(), path.name().to_string());
                    return Ok(ActorRef::remote(actor_id, member.addr, transport));
                }
            }
            return Err(anyhow::anyhow!(
                "Named actor instance not found: {}@{}",
                path_key,
                target_node
            ));
        }

        // No instance specified - check local first, then load balance
        if let Some(actor_name) = self.named_actor_paths.get(&path_key) {
            if let Some(handle) = self.local_actors.get(actor_name.value()) {
                let actor_id = ActorId::new(self.node_id.clone(), actor_name.clone());
                return Ok(ActorRef::local(actor_id, handle.sender.clone()));
            }
        }

        // Look up in cluster registry and select an instance
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            if let Some(member) = cluster.select_named_actor_instance(path).await {
                let transport = Arc::new(HttpRemoteTransport::new_named(
                    self.transport.clone(),
                    member.addr,
                    path.clone(),
                ));
                let actor_id = ActorId::new(member.node_id.clone(), path.name().to_string());
                return Ok(ActorRef::remote(actor_id, member.addr, transport));
            }
        }

        Err(anyhow::anyhow!("Named actor not found: {}", path_key))
    }

    /// Resolve a global actor address
    async fn resolve_global(&self, node_id: &NodeId, actor_id: &str) -> anyhow::Result<ActorRef> {
        // Handle localhost
        let resolved_node = if node_id.as_str() == LOCALHOST {
            &self.node_id
        } else {
            node_id
        };

        // Check local
        if resolved_node == &self.node_id {
            if let Some(handle) = self.local_actors.get(actor_id) {
                let full_actor_id = ActorId::new(self.node_id.clone(), actor_id.to_string());
                return Ok(ActorRef::local(full_actor_id, handle.sender.clone()));
            }
            return Err(anyhow::anyhow!("Local actor not found: {}", actor_id));
        }

        // Remote
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            if let Some(member) = cluster.get_member(resolved_node).await {
                let transport = Arc::new(HttpRemoteTransport::new(
                    self.transport.clone(),
                    member.addr,
                    actor_id.to_string(),
                ));
                let full_actor_id = ActorId::new(resolved_node.clone(), actor_id.to_string());
                return Ok(ActorRef::remote(full_actor_id, member.addr, transport));
            }
        }

        Err(anyhow::anyhow!(
            "Actor not found: actor://{}/{}",
            node_id,
            actor_id
        ))
    }

    /// Look up named actor info from the cluster registry
    pub async fn lookup_named(&self, path: &ActorPath) -> Option<NamedActorInfo> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            cluster.lookup_named_actor(path).await
        } else {
            None
        }
    }

    /// Get all instances of a named actor
    pub async fn get_named_instances(&self, path: &ActorPath) -> Vec<MemberInfo> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            cluster.get_named_actor_instances(path).await
        } else {
            Vec::new()
        }
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

    /// Stop an actor by name
    pub async fn stop(&self, actor_name: &str) -> anyhow::Result<()> {
        self.stop_with_reason(actor_name, StopReason::Killed).await
    }

    /// Stop an actor with a specific reason
    pub async fn stop_with_reason(
        &self,
        actor_name: &str,
        reason: StopReason,
    ) -> anyhow::Result<()> {
        if let Some((_, handle)) = self.local_actors.remove(actor_name) {
            handle.join_handle.abort();

            let local_actors = self.local_actors.clone();
            self.lifecycle
                .handle_termination(
                    &handle.actor_id,
                    actor_name,
                    handle.named_path,
                    reason,
                    &self.named_actor_paths,
                    &self.cluster,
                    |name| local_actors.get(name).map(|h| h.sender.clone()),
                )
                .await;
        }
        Ok(())
    }

    /// Handle actor termination - called when an actor naturally terminates
    async fn handle_actor_terminated(
        &self,
        actor_name: &str,
        actor_id: &ActorId,
        named_path: Option<ActorPath>,
        reason: StopReason,
    ) {
        // Only process if actor is still registered (not already stopped via stop())
        if self.local_actors.remove(actor_name).is_none() {
            return;
        }

        let local_actors = self.local_actors.clone();
        self.lifecycle
            .handle_termination(
                actor_id,
                actor_name,
                named_path,
                reason,
                &self.named_actor_paths,
                &self.cluster,
                |name| local_actors.get(name).map(|h| h.sender.clone()),
            )
            .await;
    }

    /// Stop a named actor by path
    pub async fn stop_named(&self, path: &ActorPath) -> anyhow::Result<()> {
        self.stop_named_with_reason(path, StopReason::Killed).await
    }

    /// Stop a named actor by path with a specific reason
    pub async fn stop_named_with_reason(
        &self,
        path: &ActorPath,
        reason: StopReason,
    ) -> anyhow::Result<()> {
        let path_key = path.as_str();

        // Find the local actor name for this path
        if let Some(actor_name_ref) = self.named_actor_paths.get(&path_key) {
            let actor_name = actor_name_ref.clone();
            drop(actor_name_ref);

            if let Some((_, handle)) = self.local_actors.remove(&actor_name) {
                handle.join_handle.abort();

                let local_actors = self.local_actors.clone();
                self.lifecycle
                    .handle_termination(
                        &handle.actor_id,
                        &actor_name,
                        Some(path.clone()),
                        reason,
                        &self.named_actor_paths,
                        &self.cluster,
                        |name| local_actors.get(name).map(|h| h.sender.clone()),
                    )
                    .await;
            }
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

    async fn watch(&self, watcher: &ActorId, target: &ActorId) -> anyhow::Result<()> {
        // Only support local watching for now
        if target.node != self.node_id {
            return Err(anyhow::anyhow!(
                "Cannot watch remote actor: {} (watching remote actors not yet supported)",
                target
            ));
        }

        self.lifecycle.watch(&watcher.name, &target.name).await;
        Ok(())
    }

    async fn unwatch(&self, watcher: &ActorId, target: &ActorId) -> anyhow::Result<()> {
        self.lifecycle.unwatch(&watcher.name, &target.name).await;
        Ok(())
    }
}

/// Actor message loop - returns the reason for stopping
async fn run_actor_loop<A: Actor>(
    mut actor: A,
    mut receiver: mpsc::Receiver<Envelope>,
    mut ctx: ActorContext,
    cancel: CancellationToken,
    stats: Arc<ActorStats>,
) -> StopReason {
    let stop_reason = loop {
        tokio::select! {
            msg = receiver.recv() => {
                match msg {
                    Some(envelope) => {
                        let raw = envelope.to_raw_message();

                        stats.inc_message();
                        match actor.receive(raw, &mut ctx).await {
                            Ok(response) => {
                                envelope.respond(Ok(response.payload));
                            }
                            Err(e) => {
                                tracing::error!(
                                    actor_id = ?ctx.actor_id(),
                                    error = %e,
                                    "Actor handler error"
                                );
                                envelope.respond(Err(anyhow::anyhow!("Handler error: {}", e)));
                            }
                        }
                    }
                    None => {
                        // Mailbox closed (all senders dropped)
                        break StopReason::Normal;
                    }
                }
            }
            _ = cancel.cancelled() => {
                break StopReason::SystemShutdown;
            }
        }
    };

    // Cleanup
    stats.inc_stop();
    if let Err(e) = actor.on_stop(&mut ctx).await {
        tracing::warn!(actor_id = ?ctx.actor_id(), error = %e, "Actor stop error");
        // If on_stop fails, mark as failed
        if matches!(stop_reason, StopReason::Normal) {
            return StopReason::Failed(e.to_string());
        }
    }

    stop_reason
}

/// Unified message handler for HTTP transport
struct SystemMessageHandler {
    node_id: NodeId,
    local_actors: Arc<DashMap<String, LocalActorHandle>>,
    named_actor_paths: Arc<DashMap<String, String>>,
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

        rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?
    }

    async fn handle_named_actor_message(
        &self,
        path: &str,
        msg: RawMessage,
    ) -> anyhow::Result<Vec<u8>> {
        // Look up the local actor name for this path
        let actor_name = self
            .named_actor_paths
            .get(path)
            .ok_or_else(|| anyhow::anyhow!("Named actor not found: {}", path))?
            .clone();

        let handle = self
            .local_actors
            .get(&actor_name)
            .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", actor_name))?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let envelope = Envelope::ask(msg.msg_type, msg.payload, tx);

        handle
            .sender
            .send(envelope)
            .await
            .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

        rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?
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

            let mut actor_info = serde_json::json!({
                "name": name,
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
            "actors": actors,
            "named_actors": named_actors,
            "cluster": cluster_info,
        })
    }

    async fn get_actor_info(&self, actor_name: &str) -> Option<serde_json::Value> {
        if let Some(handle) = self.local_actors.get(actor_name) {
            let mut info = serde_json::json!({
                "name": actor_name,
                "node_id": self.node_id.to_string(),
                "status": "local",
                "stats": handle.stats.to_json(),
                "metadata": handle.metadata,
            });

            if let Some(path) = &handle.named_path {
                info["named_path"] = serde_json::json!(path.as_str());
            }

            return Some(info);
        }

        None
    }

    async fn get_named_actor_info(&self, path: &str) -> Option<serde_json::Value> {
        // Check local first
        if let Some(actor_name) = self.named_actor_paths.get(path) {
            if let Some(handle) = self.local_actors.get(actor_name.value()) {
                return Some(serde_json::json!({
                    "path": path,
                    "actor_name": actor_name.value(),
                    "node_id": self.node_id.to_string(),
                    "status": "local",
                    "stats": handle.stats.to_json(),
                    "metadata": handle.metadata,
                }));
            }
        }

        // Check cluster registry
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            if let Ok(actor_path) = ActorPath::new(path) {
                if let Some(info) = cluster.lookup_named_actor(&actor_path).await {
                    return Some(serde_json::json!({
                        "path": path,
                        "instances": info.instances.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
                        "instance_count": info.instance_count(),
                        "status": "remote",
                    }));
                }
            }
        }

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
