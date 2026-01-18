//! Actor System - the main entry point for creating and managing actors
//!
//! This module provides:
//! - [`ActorSystem`] - The main system for managing actors
//! - [`SystemConfig`] - Configuration for the actor system
//! - [`SpawnOptions`] - Options for spawning actors
//! - [`ResolveOptions`] - Options for resolving named actors

mod config;
mod handle;
mod handler;
mod runtime;

pub use config::{ActorSystemBuilder, ResolveOptions, SpawnOptions, SystemConfig};
pub use handle::ActorStats;

use crate::actor::{
    Actor, ActorAddress, ActorContext, ActorId, ActorPath, ActorRef, ActorResolver, ActorSystemRef,
    IntoActorPath, Mailbox, NodeId, StopReason,
};
use crate::cluster::{
    GossipBackend, HeadNodeBackend, MemberInfo, MemberStatus, NamedActorInfo, NamingBackend,
};
use crate::policies::{LoadBalancingPolicy, RoundRobinPolicy, Worker};
use crate::system_actor::{
    BoxedActorFactory, SystemActor, SystemRef, SYSTEM_ACTOR_LOCAL_NAME, SYSTEM_ACTOR_PATH,
};
use crate::transport::{Http2RemoteTransport, Http2Transport};
use crate::watch::ActorLifecycle;
use dashmap::DashMap;
use handle::LocalActorHandle;
use handler::SystemMessageHandler;
use runtime::run_supervision_loop;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Per-node load tracking
#[derive(Debug, Default)]
pub struct NodeLoadTracker {
    /// Current in-flight requests to this node
    load: AtomicUsize,
    /// Total requests processed
    processed: AtomicU64,
}

impl NodeLoadTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&self) -> usize {
        self.load.load(Ordering::Relaxed)
    }

    pub fn increment(&self) {
        self.load.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement(&self) {
        self.load.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn processed(&self) -> u64 {
        self.processed.load(Ordering::Relaxed)
    }

    pub fn increment_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
    }
}

/// Wrapper to adapt MemberInfo to the Worker trait for load balancing
#[derive(Debug)]
struct MemberWorker {
    url: String,
    is_alive: bool,
    /// Shared load tracker for this node
    load_tracker: Arc<NodeLoadTracker>,
}

impl MemberWorker {
    fn new(member: &MemberInfo, load_tracker: Arc<NodeLoadTracker>) -> Self {
        Self {
            url: member.addr.to_string(),
            is_alive: member.status == MemberStatus::Alive,
            load_tracker,
        }
    }
}

impl Worker for MemberWorker {
    fn url(&self) -> &str {
        &self.url
    }

    fn is_healthy(&self) -> bool {
        self.is_alive
    }

    fn set_healthy(&mut self, healthy: bool) {
        self.is_alive = healthy;
    }

    fn load(&self) -> usize {
        self.load_tracker.load()
    }

    fn increment_load(&self) {
        self.load_tracker.increment();
    }

    fn decrement_load(&self) {
        self.load_tracker.decrement();
    }

    fn increment_processed(&self) {
        self.load_tracker.increment_processed();
    }

    fn processed(&self) -> u64 {
        self.load_tracker.processed()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
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

    /// Naming backend (for discovery)
    cluster: Arc<RwLock<Option<Arc<dyn NamingBackend>>>>,

    /// HTTP/2 transport
    transport: Arc<Http2Transport>,

    /// Cancellation token
    cancel_token: CancellationToken,

    /// Actor lifecycle manager (watch, termination handling)
    lifecycle: Arc<ActorLifecycle>,

    /// Actor ID counter (for generating unique local IDs)
    actor_id_counter: AtomicU64,

    /// Default load balancing policy
    default_lb_policy: Arc<dyn LoadBalancingPolicy>,

    /// Per-node load tracking for remote nodes
    node_load: Arc<DashMap<SocketAddr, Arc<NodeLoadTracker>>>,
}

impl ActorSystem {
    /// Create a builder for configuring ActorSystem
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let system = ActorSystem::builder().build().await?;
    /// ```
    pub fn builder() -> ActorSystemBuilder {
        ActorSystemBuilder::new()
    }

    /// Create a new actor system
    pub async fn new(config: SystemConfig) -> anyhow::Result<Arc<Self>> {
        let cancel_token = CancellationToken::new();
        let node_id = NodeId::generate();
        let local_actors: Arc<DashMap<String, LocalActorHandle>> = Arc::new(DashMap::new());
        let named_actor_paths: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
        let cluster_holder: Arc<RwLock<Option<Arc<dyn NamingBackend>>>> =
            Arc::new(RwLock::new(None));
        let lifecycle = Arc::new(ActorLifecycle::new());

        // Create message handler (needs cluster reference for gossip)
        let handler = SystemMessageHandler::new(
            node_id,
            local_actors.clone(),
            named_actor_paths.clone(),
            cluster_holder.clone(),
        );

        // Clone http2_config before moving it to transport
        let http2_config_for_backend = config.http2_config.clone();

        // Create HTTP/2 transport
        let (transport, actual_addr) = Http2Transport::new(
            config.addr,
            Arc::new(handler),
            config.http2_config,
            cancel_token.clone(),
        )
        .await?;

        // Create naming backend based on configuration
        let backend: Arc<dyn NamingBackend> = if config.head_addr.is_some() || config.is_head_node {
            // Head node mode: create HeadNodeBackend
            let head_config = config.head_node_config.unwrap_or_default();
            let backend = HeadNodeBackend::with_config(
                node_id,
                actual_addr,
                config.is_head_node,
                config.head_addr,
                http2_config_for_backend,
                head_config,
            );
            Arc::new(backend)
        } else {
            // Gossip mode: create GossipBackend
            let backend = GossipBackend::new(
                node_id,
                actual_addr,
                transport.clone(),
                config.gossip_config,
            );
            Arc::new(backend)
        };

        {
            let mut holder = cluster_holder.write().await;
            *holder = Some(backend.clone());
        }

        // Start backend
        backend.start(cancel_token.clone());

        // Join cluster if seed nodes provided (only for gossip mode)
        if !config.seed_nodes.is_empty() && config.head_addr.is_none() && !config.is_head_node {
            backend.join(config.seed_nodes).await?;
        } else if config.head_addr.is_some() || config.is_head_node {
            // For head node mode, join is handled internally
            backend.join(Vec::new()).await?;
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
            default_lb_policy: Arc::new(RoundRobinPolicy::new()),
            node_load: Arc::new(DashMap::new()),
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

    /// Get a local actor reference by name
    ///
    /// Returns None if the actor doesn't exist locally
    pub fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef> {
        self.local_actors
            .get(name)
            .map(|handle| ActorRef::local(handle.actor_id, handle.sender.clone()))
    }

    /// Generate a new unique local actor ID
    fn next_actor_id(&self) -> ActorId {
        let local_id = self.actor_id_counter.fetch_add(1, Ordering::Relaxed);
        ActorId::new(self.node_id, local_id)
    }

    // ========== Spawn Methods ==========

    /// Create a once-use factory from an actor instance
    fn once_factory<A: Actor>(actor: A) -> impl FnMut() -> anyhow::Result<A> {
        let mut actor_opt = Some(actor);
        move || {
            actor_opt
                .take()
                .ok_or_else(|| anyhow::anyhow!("Actor cannot be restarted (spawned as instance)"))
        }
    }

    /// Spawn an actor with a local name (uses system default mailbox capacity)
    pub async fn spawn<A>(
        self: &Arc<Self>,
        name: impl AsRef<str>,
        actor: A,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        self.spawn_factory(name, Self::once_factory(actor), SpawnOptions::default())
            .await
    }

    /// Spawn an actor with custom options
    pub async fn spawn_with_options<A>(
        self: &Arc<Self>,
        name: impl AsRef<str>,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        self.spawn_factory(name, Self::once_factory(actor), options)
            .await
    }

    /// Spawn an actor using a factory function (enables supervision restarts)
    pub async fn spawn_factory<F, A>(
        self: &Arc<Self>,
        name: impl AsRef<str>,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
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
        let metadata = HashMap::new();

        // Create context with system reference for actor_ref/watch/schedule_self
        let ctx = ActorContext::with_system(
            actor_id,
            self.clone() as Arc<dyn ActorSystemRef>,
            self.cancel_token.clone(),
            sender.clone(),
        );

        // Spawn actor loop
        let stats_clone = stats.clone();
        let cancel = self.cancel_token.clone();
        let actor_id_for_log = actor_id;
        let supervision = options.supervision.clone();

        let join_handle = tokio::spawn(async move {
            let reason =
                run_supervision_loop(factory, receiver, ctx, cancel, stats_clone, supervision)
                    .await;
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
    ///
    /// # Example
    /// ```rust,ignore
    /// // Path can be &str, String, or ActorPath
    /// system.spawn_named("services/echo", "echo", MyActor).await?;
    /// ```
    pub async fn spawn_named<P, A>(
        self: &Arc<Self>,
        path: P,
        local_name: impl AsRef<str>,
        actor: A,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        A: Actor,
    {
        let path = path.into_actor_path()?;
        self.spawn_named_factory(
            path,
            local_name,
            Self::once_factory(actor),
            SpawnOptions::default(),
        )
        .await
    }

    /// Spawn a named actor with custom options
    pub async fn spawn_named_with_options<P, A>(
        self: &Arc<Self>,
        path: P,
        local_name: impl AsRef<str>,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        A: Actor,
    {
        let path = path.into_actor_path()?;
        self.spawn_named_factory(path, local_name, Self::once_factory(actor), options)
            .await
    }

    /// Spawn a named actor using a factory function
    pub async fn spawn_named_factory<P, F, A>(
        self: &Arc<Self>,
        path: P,
        local_name: impl AsRef<str>,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor,
    {
        let path = path.into_actor_path()?;
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
        let metadata = options.metadata.clone();

        // Create context with system reference for actor_ref/watch/schedule_self
        let ctx = ActorContext::with_system(
            actor_id,
            self.clone() as Arc<dyn ActorSystemRef>,
            self.cancel_token.clone(),
            sender.clone(),
        );

        // Spawn actor loop
        let stats_clone = stats.clone();
        let cancel = self.cancel_token.clone();
        let actor_id_for_log = actor_id;
        let supervision = options.supervision.clone();

        let join_handle = tokio::spawn(async move {
            let reason =
                run_supervision_loop(factory, receiver, ctx, cancel, stats_clone, supervision)
                    .await;
            tracing::debug!(actor_id = ?actor_id_for_log, reason = ?reason, "Actor stopped");
        });

        // Register actor
        let handle = LocalActorHandle {
            sender: sender.clone(),
            join_handle,
            stats: stats.clone(),
            metadata: metadata.clone(),
            named_path: Some(path.clone()),
            actor_id,
        };

        self.local_actors.insert(local_name.to_string(), handle);
        self.named_actor_paths
            .insert(path.as_str().to_string(), local_name.to_string());

        // Register in cluster with full details
        if let Some(cluster) = self.cluster.read().await.as_ref() {
            if metadata.is_empty() {
                cluster.register_named_actor(path.clone()).await;
            } else {
                cluster
                    .register_named_actor_full(path.clone(), actor_id, metadata)
                    .await;
            }
        }

        // Create ActorRef
        Ok(ActorRef::local(actor_id, sender))
    }

    // ========== Resolve Methods ==========

    /// Get ActorRef for a local or remote actor by ID
    pub async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef> {
        // Check if local
        if id.node() == self.node_id || id.node().is_local() {
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

    /// Resolve a named actor by path (direct resolution)
    ///
    /// Returns an ActorRef that points to the current location of the named actor.
    /// Note: If the actor migrates, this reference may become stale.
    /// For actors that may migrate, consider using `resolve_named_lazy`.
    ///
    /// # Example
    /// ```rust,ignore
    /// let actor = system.resolve_named("services/echo", None).await?;
    /// ```
    pub async fn resolve_named<P>(
        &self,
        path: P,
        node_id: Option<&NodeId>,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
    {
        let path = path.into_actor_path()?;
        let options = if let Some(nid) = node_id {
            ResolveOptions::new().node_id(*nid)
        } else {
            ResolveOptions::new()
        };
        self.resolve_named_with_options(&path, options).await
    }

    /// Resolve a named actor with lazy resolution (re-resolves after cache expires)
    ///
    /// Returns an ActorRef that automatically re-resolves after 5 seconds.
    /// This is useful for named actors that may migrate between nodes.
    ///
    /// # Example
    /// ```rust,ignore
    /// let actor = system.resolve_named_lazy("services/echo").await?;
    /// // Even if the actor migrates, this ref will find it after cache expires
    /// ```
    pub fn resolve_named_lazy<P>(self: &Arc<Self>, path: P) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
    {
        let path = path.into_actor_path()?;
        Ok(ActorRef::lazy(path, self.clone() as Arc<dyn ActorResolver>))
    }

    /// Internal: Direct resolution for ActorResolver trait
    async fn resolve_named_direct(
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
            // Use load balancing policy
            let policy = options.policy.as_ref().unwrap_or(&self.default_lb_policy);
            self.select_instance(&healthy_instances, policy.as_ref())
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

        let actor_id = ActorId::new(target.node_id, 0);
        Ok(ActorRef::remote(actor_id, target.addr, Arc::new(transport)))
    }

    /// Select an instance using load balancing policy
    fn select_instance<'a>(
        &self,
        instances: &'a [MemberInfo],
        policy: &dyn LoadBalancingPolicy,
    ) -> &'a MemberInfo {
        // Convert MemberInfo to Worker wrappers with load tracking
        let workers: Vec<Arc<dyn crate::policies::Worker>> = instances
            .iter()
            .map(|m| {
                // Get or create load tracker for this node
                let tracker = self
                    .node_load
                    .entry(m.addr)
                    .or_insert_with(|| Arc::new(NodeLoadTracker::new()))
                    .clone();
                Arc::new(MemberWorker::new(m, tracker)) as Arc<dyn crate::policies::Worker>
            })
            .collect();

        // Use the policy to select a worker index
        let idx = policy.select_worker(&workers, None).unwrap_or(0);

        // Increment load for selected node (caller should decrement after request completes)
        if let Some(tracker) = self.node_load.get(&instances[idx].addr) {
            tracker.increment();
        }

        &instances[idx]
    }

    /// Get load tracker for a node address
    pub fn get_node_load_tracker(&self, addr: &SocketAddr) -> Option<Arc<NodeLoadTracker>> {
        self.node_load.get(addr).map(|r| r.clone())
    }

    /// Decrement load after a request completes
    pub fn decrement_node_load(&self, addr: &SocketAddr) {
        if let Some(tracker) = self.node_load.get(addr) {
            tracker.decrement();
            tracker.increment_processed();
        }
    }

    /// Resolve an actor address and get an ActorRef
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

    /// Get detailed instances with actor_id and metadata
    pub async fn get_named_instances_detailed(
        &self,
        path: &ActorPath,
    ) -> Vec<(MemberInfo, Option<crate::cluster::NamedActorInstance>)> {
        let cluster_guard = self.cluster.read().await;
        if let Some(cluster) = cluster_guard.as_ref() {
            cluster.get_named_actor_instances_detailed(path).await
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

    // ========== Stop Methods ==========

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
    ///
    /// This method performs a graceful shutdown:
    /// 1. Signals cancellation to all actors
    /// 2. Triggers lifecycle cleanup for each actor (watch notifications, cluster broadcast, etc.)
    /// 3. Leaves the cluster gracefully
    /// 4. Clears all actors and watch relationships
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        tracing::info!("Shutting down actor system");

        // Signal cancellation first - this tells actors to stop processing new messages
        self.cancel_token.cancel();

        // Collect all actor info before processing to avoid holding locks during cleanup
        let actor_entries: Vec<_> = self
            .local_actors
            .iter()
            .map(|entry| {
                (
                    entry.key().clone(),
                    entry.actor_id,
                    entry.named_path.clone(),
                    entry.join_handle.abort_handle(),
                )
            })
            .collect();

        // Process each actor's termination with proper lifecycle handling
        for (actor_name, actor_id, named_path, abort_handle) in actor_entries {
            // Abort the actor task
            abort_handle.abort();

            // Trigger lifecycle cleanup (watch notifications, cluster broadcast, routing cleanup)
            let local_actors = self.local_actors.clone();
            self.lifecycle
                .handle_termination(
                    &actor_id,
                    &actor_name,
                    named_path,
                    StopReason::SystemShutdown,
                    &self.named_actor_paths,
                    &self.cluster,
                    |name| local_actors.get(name).map(|h| h.sender.clone()),
                )
                .await;
        }

        // Clear all actors
        self.local_actors.clear();

        // Clear all watch relationships
        self.lifecycle.clear().await;

        // Leave cluster gracefully
        {
            let cluster_guard = self.cluster.read().await;
            if let Some(cluster) = cluster_guard.as_ref() {
                cluster.leave().await?;
            }
        }

        tracing::info!("Actor system shutdown complete");
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

/// Implement ActorResolver for ActorSystem
///
/// This enables lazy ActorRef to resolve named actors on demand.
#[async_trait::async_trait]
impl ActorResolver for ActorSystem {
    async fn resolve_path(&self, path: &ActorPath) -> anyhow::Result<ActorRef> {
        // Use direct resolution (not lazy) to avoid infinite recursion
        self.resolve_named_direct(path, None).await
    }
}
