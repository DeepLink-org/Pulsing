//! Actor System - the main entry point for creating and managing actors

use crate::actor::{
    Actor, ActorAddress, ActorContext, ActorId, ActorPath, ActorRef, ActorSystemRef, Envelope,
    Mailbox, Message, NodeId, StopReason, DEFAULT_MAILBOX_SIZE,
};
use crate::cluster::{
    GossipCluster, GossipConfig, GossipMessage, MemberInfo, MemberStatus, NamedActorInfo,
};
use crate::metrics::{metrics, SystemMetrics as PrometheusMetrics};
use crate::system_actor::{
    BoxedActorFactory, SystemActor, SystemRef, SYSTEM_ACTOR_LOCAL_NAME, SYSTEM_ACTOR_PATH,
};
use crate::transport::{Http2Config, Http2RemoteTransport, Http2ServerHandler, Http2Transport};
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
    /// HTTP/2 address for all communication (actors + gossip)
    pub addr: SocketAddr,

    /// Seed nodes to join (HTTP/2 addresses)
    pub seed_nodes: Vec<SocketAddr>,

    /// Gossip configuration
    pub gossip_config: GossipConfig,

    /// HTTP/2 transport configuration
    pub http2_config: Http2Config,

    /// Default mailbox capacity for all actors
    pub default_mailbox_capacity: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            addr: "0.0.0.0:0".parse().unwrap(),
            seed_nodes: Vec::new(),
            gossip_config: GossipConfig::default(),
            http2_config: Http2Config::default(),
            default_mailbox_capacity: DEFAULT_MAILBOX_SIZE,
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

    /// Set default mailbox capacity
    pub fn with_mailbox_capacity(mut self, capacity: usize) -> Self {
        self.default_mailbox_capacity = capacity;
        self
    }
}

/// Options for spawning an actor
#[derive(Default, Clone, Debug)]
pub struct SpawnOptions {
    /// Override mailbox capacity (None = use system default)
    pub mailbox_capacity: Option<usize>,
    /// Whether this actor is public (can be resolved by name across cluster)
    pub public: bool,
}

impl SpawnOptions {
    /// Create new spawn options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set mailbox capacity override
    pub fn mailbox_capacity(mut self, capacity: usize) -> Self {
        self.mailbox_capacity = Some(capacity);
        self
    }

    /// Set whether actor is public
    pub fn public(mut self, public: bool) -> Self {
        self.public = public;
        self
    }
}

/// Load balance strategy for resolving named actors with multiple instances
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Pick the first available instance (original behavior)
    First,
    /// Round-robin across instances
    #[default]
    RoundRobin,
    /// Random selection
    Random,
    /// Prefer local instance if available, fallback to round-robin
    PreferLocal,
}

/// Options for resolving named actors
#[derive(Clone, Debug, Default)]
pub struct ResolveOptions {
    /// Target node ID (if specified, skip load balancing)
    pub node_id: Option<NodeId>,
    /// Load balance strategy (default: RoundRobin)
    pub strategy: LoadBalanceStrategy,
    /// Only select Alive nodes (default: true)
    pub filter_alive: bool,
}

impl ResolveOptions {
    /// Create new resolve options with defaults
    pub fn new() -> Self {
        Self {
            filter_alive: true,
            ..Default::default()
        }
    }

    /// Set target node ID (bypasses load balancing)
    pub fn node_id(mut self, node_id: NodeId) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Set load balance strategy
    pub fn strategy(mut self, strategy: LoadBalanceStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set whether to filter only alive nodes
    pub fn filter_alive(mut self, filter: bool) -> Self {
        self.filter_alive = filter;
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

    /// HTTP/2 address
    addr: SocketAddr,

    /// Default mailbox capacity for actors
    default_mailbox_capacity: usize,

    /// Local actors (actor_name -> handle)
    local_actors: Arc<DashMap<String, LocalActorHandle>>,

    /// Named actor path to local actor name mapping (path_string -> actor_name)
    named_actor_paths: Arc<DashMap<String, String>>,

    /// Gossip cluster (for discovery)
    cluster: Arc<RwLock<Option<Arc<GossipCluster>>>>,

    /// HTTP/2 transport
    transport: Arc<Http2Transport>,

    /// Cancellation token
    cancel_token: CancellationToken,

    /// Actor lifecycle manager (watch, termination handling)
    lifecycle: Arc<ActorLifecycle>,

    /// Actor ID counter (for generating unique local IDs)
    actor_id_counter: AtomicU64,

    /// Round-robin counter for load balancing (per actor path)
    lb_counters: DashMap<String, AtomicU64>,
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
            node_id,
            local_actors: local_actors.clone(),
            named_actor_paths: named_actor_paths.clone(),
            cluster: cluster_holder.clone(),
        };

        // Create HTTP/2 transport
        let (transport, actual_addr) = Http2Transport::new(
            config.addr,
            Arc::new(handler),
            config.http2_config,
            cancel_token.clone(),
        )
        .await?;

        // Create gossip cluster
        let cluster = GossipCluster::new(
            node_id,
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
            default_mailbox_capacity: config.default_mailbox_capacity,
            local_actors: local_actors.clone(),
            named_actor_paths: named_actor_paths.clone(),
            cluster: cluster_holder,
            transport,
            cancel_token: cancel_token.clone(),
            lifecycle,
            actor_id_counter: AtomicU64::new(1),
            lb_counters: DashMap::new(),
        });

        // Start the builtin SystemActor with path "system"
        system
            .start_system_actor(local_actors, named_actor_paths)
            .await?;

        tracing::info!(
            node_id = %node_id,
            addr = %actual_addr,
            "Actor system started"
        );

        Ok(system)
    }

    /// Start the builtin SystemActor
    async fn start_system_actor(
        self: &Arc<Self>,
        local_actors: Arc<DashMap<String, LocalActorHandle>>,
        named_actor_paths: Arc<DashMap<String, String>>,
    ) -> anyhow::Result<()> {
        // Create SystemRef for SystemActor
        let system_ref = Arc::new(SystemRef {
            node_id: self.node_id,
            addr: self.addr,
            local_actors: local_actors
                .iter()
                .map(|e| (e.key().clone(), e.sender.clone()))
                .collect::<DashMap<_, _>>()
                .into(),
            named_actor_paths,
        });

        // Create SystemActor with default factory
        let system_actor = SystemActor::with_default_factory(system_ref);

        // Spawn as named actor with path "system"
        let path = ActorPath::new(SYSTEM_ACTOR_PATH)?;
        self.spawn_named(path, SYSTEM_ACTOR_LOCAL_NAME, system_actor)
            .await?;

        tracing::debug!(path = SYSTEM_ACTOR_PATH, "SystemActor started");
        Ok(())
    }

    /// Start SystemActor with custom factory (for Python extension)
    pub async fn start_system_actor_with_factory(
        self: &Arc<Self>,
        factory: BoxedActorFactory,
    ) -> anyhow::Result<()> {
        // Check if already started
        if self.local_actors.contains_key(SYSTEM_ACTOR_LOCAL_NAME) {
            return Err(anyhow::anyhow!("SystemActor already started"));
        }

        // Create SystemRef
        let system_ref = Arc::new(SystemRef {
            node_id: self.node_id,
            addr: self.addr,
            local_actors: Arc::new(DashMap::new()), // Will be updated
            named_actor_paths: self.named_actor_paths.clone(),
        });

        // Create SystemActor with custom factory
        let system_actor = SystemActor::new(system_ref, factory);

        // Spawn as named actor
        let path = ActorPath::new(SYSTEM_ACTOR_PATH)?;
        self.spawn_named(path, SYSTEM_ACTOR_LOCAL_NAME, system_actor)
            .await?;

        tracing::debug!(
            path = SYSTEM_ACTOR_PATH,
            "SystemActor started with custom factory"
        );
        Ok(())
    }

    /// Get SystemActor reference
    pub async fn system(&self) -> anyhow::Result<ActorRef> {
        self.resolve_named(&ActorPath::new(SYSTEM_ACTOR_PATH)?, None)
            .await
    }

    /// Get node ID
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Get local address
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Get list of local actor names
    pub fn local_actor_names(&self) -> Vec<String> {
        self.local_actors.iter().map(|e| e.key().clone()).collect()
    }

    /// Generate a new unique local actor ID
    fn next_actor_id(&self) -> ActorId {
        let local_id = self.actor_id_counter.fetch_add(1, Ordering::Relaxed);
        ActorId::new(self.node_id, local_id)
    }

    /// Spawn an actor with a local name (uses system default mailbox capacity)
    pub async fn spawn<A>(&self, name: impl AsRef<str>, actor: A) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        self.spawn_with_options(name, actor, SpawnOptions::default())
            .await
    }

    /// Spawn an actor with custom options
    pub async fn spawn_with_options<A>(
        &self,
        name: impl AsRef<str>,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        let name = name.as_ref();

        // Check for duplicate
        if self.local_actors.contains_key(name) {
            return Err(anyhow::anyhow!("Actor already exists: {}", name));
        }

        let actor_id = self.next_actor_id();

        // Use configured mailbox capacity
        let capacity = options
            .mailbox_capacity
            .unwrap_or(self.default_mailbox_capacity);
        let mailbox = Mailbox::with_capacity(capacity);
        let (sender, receiver) = mailbox.split();

        let stats = Arc::new(ActorStats::default());
        let metadata = actor.metadata();

        // Create context
        let ctx = ActorContext::new(actor_id);

        // Spawn actor loop
        let stats_clone = stats.clone();
        let cancel = self.cancel_token.clone();
        let actor_id_for_log = actor_id;
        let join_handle = tokio::spawn(async move {
            let reason = run_actor_loop(actor, receiver, ctx, cancel, stats_clone).await;
            tracing::debug!(actor_id = ?actor_id_for_log, reason = ?reason, "Actor stopped");
        });

        // Register actor
        let handle = LocalActorHandle {
            sender: sender.clone(),
            join_handle,
            stats: stats.clone(),
            metadata,
            named_path: None,
            actor_id,
        };

        self.local_actors.insert(name.to_string(), handle);

        // Create ActorRef
        Ok(ActorRef::local(actor_id, sender))
    }

    /// Spawn a named actor (publicly accessible via named path)
    pub async fn spawn_named<A>(
        &self,
        path: ActorPath,
        local_name: impl AsRef<str>,
        actor: A,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        self.spawn_named_with_options(path, local_name, actor, SpawnOptions::default())
            .await
    }

    /// Spawn a named actor with custom options
    pub async fn spawn_named_with_options<A>(
        &self,
        path: ActorPath,
        local_name: impl AsRef<str>,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        let local_name = local_name.as_ref();

        // Check for duplicate local name
        if self.local_actors.contains_key(local_name) {
            return Err(anyhow::anyhow!("Actor already exists: {}", local_name));
        }

        // Check for duplicate named path
        if self
            .named_actor_paths
            .contains_key(&path.as_str().to_string())
        {
            return Err(anyhow::anyhow!(
                "Named path already registered: {}",
                path.as_str()
            ));
        }

        let actor_id = self.next_actor_id();

        // Use configured mailbox capacity
        let capacity = options
            .mailbox_capacity
            .unwrap_or(self.default_mailbox_capacity);
        let mailbox = Mailbox::with_capacity(capacity);
        let (sender, receiver) = mailbox.split();

        let stats = Arc::new(ActorStats::default());
        let metadata = actor.metadata();

        // Create context
        let ctx = ActorContext::new(actor_id);

        // Spawn actor loop
        let stats_clone = stats.clone();
        let cancel = self.cancel_token.clone();
        let actor_id_for_log = actor_id;

        let join_handle = tokio::spawn(async move {
            let reason = run_actor_loop(actor, receiver, ctx, cancel, stats_clone).await;
            tracing::debug!(actor_id = ?actor_id_for_log, reason = ?reason, "Actor stopped");
        });

        // Register actor
        let handle = LocalActorHandle {
            sender: sender.clone(),
            join_handle,
            stats: stats.clone(),
            metadata,
            named_path: Some(path.clone()),
            actor_id,
        };

        self.local_actors.insert(local_name.to_string(), handle);
        self.named_actor_paths
            .insert(path.as_str().to_string(), local_name.to_string());

        // Register in cluster
        if let Some(cluster) = self.cluster.read().await.as_ref() {
            cluster.register_named_actor(path.clone()).await;
        }

        // Create ActorRef
        Ok(ActorRef::local(actor_id, sender))
    }

    /// Get ActorRef for a local or remote actor by ID
    pub async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef> {
        // Check if local
        if id.node() == self.node_id || id.node().is_local() {
            // Find local actor by iterating (since we don't have name in ActorId anymore)
            // When id.node().is_local(), compare only local_id since node_id=0 means "current node"
            let target_local_id = id.local_id();
            for entry in self.local_actors.iter() {
                let entry_local_id = entry.value().actor_id.local_id();
                if entry_local_id == target_local_id {
                    return Ok(ActorRef::local(
                        entry.value().actor_id,
                        entry.value().sender.clone(),
                    ));
                }
            }
            return Err(anyhow::anyhow!("Local actor not found: {}", id));
        }

        // Remote actor - get address from cluster
        let cluster_guard = self.cluster.read().await;
        let cluster = cluster_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cluster not initialized"))?;

        let member = cluster
            .get_member(&id.node())
            .await
            .ok_or_else(|| anyhow::anyhow!("Node not found in cluster: {}", id.node()))?;

        // Create remote transport using actor id
        let transport = Http2RemoteTransport::new_by_id(self.transport.client(), member.addr, *id);

        Ok(ActorRef::remote(*id, member.addr, Arc::new(transport)))
    }

    /// Resolve a named actor and get an ActorRef (uses default load balancing: RoundRobin)
    pub async fn resolve_named(
        &self,
        path: &ActorPath,
        node_id: Option<&NodeId>,
    ) -> anyhow::Result<ActorRef> {
        let options = if let Some(nid) = node_id {
            ResolveOptions::new().node_id(*nid)
        } else {
            ResolveOptions::new()
        };
        self.resolve_named_with_options(path, options).await
    }

    /// Resolve a named actor with custom options (load balancing, health filtering)
    pub async fn resolve_named_with_options(
        &self,
        path: &ActorPath,
        options: ResolveOptions,
    ) -> anyhow::Result<ActorRef> {
        let cluster_guard = self.cluster.read().await;
        let cluster = cluster_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cluster not initialized"))?;

        let instances = cluster.get_named_actor_instances(path).await;

        if instances.is_empty() {
            return Err(anyhow::anyhow!("Named actor not found: {}", path.as_str()));
        }

        // Health filtering: only select Alive nodes
        let healthy_instances: Vec<_> = if options.filter_alive {
            instances
                .into_iter()
                .filter(|i| i.status == MemberStatus::Alive)
                .collect()
        } else {
            instances
        };

        if healthy_instances.is_empty() {
            return Err(anyhow::anyhow!(
                "No healthy instances for named actor: {}",
                path.as_str()
            ));
        }

        // Select target instance
        let target = if let Some(nid) = options.node_id {
            // If node_id specified, find that specific instance
            healthy_instances
                .iter()
                .find(|i| i.node_id == nid)
                .ok_or_else(|| anyhow::anyhow!("Actor instance not found on node: {}", nid))?
        } else {
            // Use load balancing strategy
            self.select_instance(path, &healthy_instances, options.strategy)
        };

        // If local, get local ref
        if target.node_id == self.node_id {
            let actor_name = self
                .named_actor_paths
                .get(&path.as_str())
                .ok_or_else(|| anyhow::anyhow!("Named actor not found locally"))?
                .clone();

            let handle = self
                .local_actors
                .get(&actor_name)
                .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", actor_name))?;

            return Ok(ActorRef::local(handle.actor_id, handle.sender.clone()));
        }

        // Remote actor
        let transport =
            Http2RemoteTransport::new_named(self.transport.client(), target.addr, path.clone());

        // For remote named actors, we use a placeholder ActorId since we don't know the actual ID
        // The transport will use the path for routing
        let actor_id = ActorId::new(target.node_id, 0);
        Ok(ActorRef::remote(actor_id, target.addr, Arc::new(transport)))
    }

    /// Select an instance based on load balancing strategy
    fn select_instance<'a>(
        &self,
        path: &ActorPath,
        instances: &'a [MemberInfo],
        strategy: LoadBalanceStrategy,
    ) -> &'a MemberInfo {
        match strategy {
            LoadBalanceStrategy::First => &instances[0],

            LoadBalanceStrategy::RoundRobin => {
                let path_key = path.as_str().to_string();
                let counter = self
                    .lb_counters
                    .entry(path_key)
                    .or_insert_with(|| AtomicU64::new(0));
                let idx = counter.fetch_add(1, Ordering::Relaxed) as usize % instances.len();
                &instances[idx]
            }

            LoadBalanceStrategy::Random => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                use std::time::SystemTime;

                // Simple random using time-based seed
                let mut hasher = DefaultHasher::new();
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
                    .hash(&mut hasher);
                std::thread::current().id().hash(&mut hasher);
                let idx = hasher.finish() as usize % instances.len();
                &instances[idx]
            }

            LoadBalanceStrategy::PreferLocal => {
                // Try to find local instance first
                if let Some(local) = instances.iter().find(|i| i.node_id == self.node_id) {
                    local
                } else {
                    // Fallback to round-robin
                    self.select_instance(path, instances, LoadBalanceStrategy::RoundRobin)
                }
            }
        }
    }

    /// Resolve an actor address and get an ActorRef
    ///
    /// This is a general resolution method that handles both Named and Global addresses.
    pub async fn resolve(&self, address: &ActorAddress) -> anyhow::Result<ActorRef> {
        match address {
            ActorAddress::Named { path, instance } => {
                self.resolve_named(path, instance.as_ref()).await
            }
            ActorAddress::Global { node_id, actor_id } => {
                let id = ActorId::new(*node_id, *actor_id);
                self.actor_ref(&id).await
            }
        }
    }

    /// Get all instances of a named actor across the cluster
    pub async fn get_named_instances(&self, path: &ActorPath) -> Vec<MemberInfo> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            cluster.get_named_actor_instances(path).await
        } else {
            Vec::new()
        }
    }

    /// Lookup named actor information
    pub async fn lookup_named(&self, path: &ActorPath) -> Option<NamedActorInfo> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            cluster.lookup_named_actor(path).await
        } else {
            None
        }
    }

    /// Get cluster member information
    pub async fn members(&self) -> Vec<MemberInfo> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            cluster.all_members().await
        } else {
            Vec::new()
        }
    }

    /// Get all named actors in the cluster
    pub async fn all_named_actors(&self) -> Vec<NamedActorInfo> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            cluster.all_named_actors().await
        } else {
            Vec::new()
        }
    }

    /// Stop an actor
    pub async fn stop(&self, name: impl AsRef<str>) -> anyhow::Result<()> {
        self.stop_with_reason(name, StopReason::Killed).await
    }

    /// Stop an actor with a specific reason
    pub async fn stop_with_reason(
        &self,
        name: impl AsRef<str>,
        reason: StopReason,
    ) -> anyhow::Result<()> {
        let name = name.as_ref();

        if let Some((_, handle)) = self.local_actors.remove(name) {
            handle.join_handle.abort();

            let local_actors = self.local_actors.clone();
            self.lifecycle
                .handle_termination(
                    &handle.actor_id,
                    name,
                    handle.named_path.clone(),
                    reason,
                    &self.named_actor_paths,
                    &self.cluster,
                    |n| local_actors.get(n).map(|h| h.sender.clone()),
                )
                .await;
        }

        Ok(())
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

    fn node_id(&self) -> NodeId {
        self.node_id
    }

    async fn watch(&self, watcher: &ActorId, target: &ActorId) -> anyhow::Result<()> {
        // Only support local watching for now
        if target.node() != self.node_id {
            return Err(anyhow::anyhow!(
                "Cannot watch remote actor: {} (watching remote actors not yet supported)",
                target
            ));
        }

        // Use string representation of ActorId for watching
        let watcher_key = watcher.to_string();
        let target_key = target.to_string();
        self.lifecycle.watch(&watcher_key, &target_key).await;
        Ok(())
    }

    async fn unwatch(&self, watcher: &ActorId, target: &ActorId) -> anyhow::Result<()> {
        let watcher_key = watcher.to_string();
        let target_key = target.to_string();
        self.lifecycle.unwatch(&watcher_key, &target_key).await;
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
    // Call on_start
    if let Err(e) = actor.on_start(&mut ctx).await {
        tracing::error!(actor_id = ?ctx.id(), error = %e, "Actor start error");
        stats.inc_stop();
        return StopReason::Failed(e.to_string());
    }

    let stop_reason = loop {
        tokio::select! {
            msg = receiver.recv() => {
                match msg {
                    Some(envelope) => {
                        stats.inc_message();
                        let (message, responder) = envelope.into_parts();

                        match actor.receive(message, &mut ctx).await {
                            Ok(response) => responder.send(Ok(response)),
                            Err(e) => {
                                tracing::error!(actor_id = ?ctx.id(), error = %e, "Actor error");
                                responder.send(Err(anyhow::anyhow!("Handler error: {}", e)));
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
        tracing::warn!(actor_id = ?ctx.id(), error = %e, "Actor stop error");
        // If on_stop fails, mark as failed
        if matches!(stop_reason, StopReason::Normal) {
            return StopReason::Failed(e.to_string());
        }
    }

    stop_reason
}

/// Unified message handler for HTTP/2 transport
struct SystemMessageHandler {
    node_id: NodeId,
    local_actors: Arc<DashMap<String, LocalActorHandle>>,
    named_actor_paths: Arc<DashMap<String, String>>,
    cluster: Arc<RwLock<Option<Arc<GossipCluster>>>>,
}

impl SystemMessageHandler {
    /// Dispatch a message to an actor (ask pattern)
    async fn dispatch_message(&self, path: &str, msg: Message) -> anyhow::Result<Message> {
        // Check if path is /actors/{name} or /named/{path}
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
        // Check if path is /actors/{name} or /named/{path}
        if let Some(actor_name) = path.strip_prefix("/actors/") {
            self.tell_local_actor(actor_name, msg).await
        } else if let Some(named_path) = path.strip_prefix("/named/") {
            self.tell_named_actor(named_path, msg).await
        } else {
            Err(anyhow::anyhow!("Invalid path: {}", path))
        }
    }

    async fn send_to_local_actor(&self, actor_name: &str, msg: Message) -> anyhow::Result<Message> {
        // Find actor sender - first try by name, then by local_id
        let sender = if let Some(handle) = self.local_actors.get(actor_name) {
            handle.sender.clone()
        } else if let Ok(local_id) = actor_name.parse::<u64>() {
            // Try to find by local_id
            self.local_actors
                .iter()
                .find(|entry| entry.value().actor_id.local_id() == local_id)
                .map(|entry| entry.value().sender.clone())
                .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", actor_name))?
        } else {
            return Err(anyhow::anyhow!("Actor not found: {}", actor_name));
        };

        let (tx, rx) = tokio::sync::oneshot::channel();
        let envelope = Envelope::ask(msg, tx);

        sender
            .send(envelope)
            .await
            .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

        rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?
    }

    async fn tell_local_actor(&self, actor_name: &str, msg: Message) -> anyhow::Result<()> {
        // Find actor sender - first try by name, then by local_id
        let sender = if let Some(handle) = self.local_actors.get(actor_name) {
            handle.sender.clone()
        } else if let Ok(local_id) = actor_name.parse::<u64>() {
            // Try to find by local_id
            self.local_actors
                .iter()
                .find(|entry| entry.value().actor_id.local_id() == local_id)
                .map(|entry| entry.value().sender.clone())
                .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", actor_name))?
        } else {
            return Err(anyhow::anyhow!("Actor not found: {}", actor_name));
        };

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
    ///
    /// This handler supports both single and streaming requests:
    /// - Single requests are dispatched to local actors
    /// - Streaming requests are passed through to actors that support streaming
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
        if let Some(cluster) = cluster_guard.as_ref() {
            let msg: GossipMessage = bincode::deserialize(&payload)?;
            let response = cluster.handle_gossip(msg, peer_addr).await?;
            if let Some(resp) = response {
                Ok(Some(bincode::serialize(&resp)?))
            } else {
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
            actors_created: self.local_actors.len() as u64, // Approximation
            actors_stopped: 0,                              // Would need separate tracking
            cluster_members,
        };

        // Export using global metrics registry
        metrics().export_prometheus(&system_metrics)
    }
}
