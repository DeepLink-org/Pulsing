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
mod traits;

pub use config::{ActorSystemBuilder, ResolveOptions, SpawnOptions, SystemConfig};
pub use handle::ActorStats;
pub use traits::{ActorSystemAdvancedExt, ActorSystemCoreExt, ActorSystemOpsExt};

use crate::actor::{
    Actor, ActorAddress, ActorContext, ActorId, ActorPath, ActorRef, ActorResolver, ActorSystemRef,
    Envelope, IntoActor, IntoActorPath, Mailbox, NodeId, StopReason,
};
use crate::cluster::{
    GossipBackend, HeadNodeBackend, MemberInfo, MemberStatus, NamedActorInfo, NamingBackend,
};
use crate::policies::{LoadBalancingPolicy, RoundRobinPolicy, Worker};
use crate::system_actor::{BoxedActorFactory, SystemActor, SystemRef, SYSTEM_ACTOR_PATH};
use crate::transport::{Http2RemoteTransport, Http2Transport};
use crate::watch::ActorLifecycle;
use dashmap::DashMap;
use handle::LocalActorHandle;
use handler::SystemMessageHandler;
use runtime::{run_actor_instance, run_supervision_loop};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Per-node load tracking with activity timestamp for cleanup
#[derive(Debug)]
pub struct NodeLoadTracker {
    /// Current in-flight requests to this node
    load: AtomicUsize,
    /// Total requests processed
    processed: AtomicU64,
    /// Last activity timestamp (Unix millis) for stale entry cleanup
    last_activity_millis: AtomicU64,
}

impl Default for NodeLoadTracker {
    fn default() -> Self {
        Self {
            load: AtomicUsize::new(0),
            processed: AtomicU64::new(0),
            last_activity_millis: AtomicU64::new(Self::current_millis()),
        }
    }
}

impl NodeLoadTracker {
    pub fn new() -> Self {
        Self::default()
    }

    fn current_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn touch(&self) {
        self.last_activity_millis
            .store(Self::current_millis(), Ordering::Relaxed);
    }

    pub fn load(&self) -> usize {
        self.load.load(Ordering::Relaxed)
    }

    pub fn increment(&self) {
        self.load.fetch_add(1, Ordering::Relaxed);
        self.touch();
    }

    pub fn decrement(&self) {
        self.load.fetch_sub(1, Ordering::Relaxed);
        self.touch();
    }

    pub fn processed(&self) -> u64 {
        self.processed.load(Ordering::Relaxed)
    }

    pub fn increment_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
        self.touch();
    }

    /// Returns elapsed time since last activity
    pub fn last_activity_elapsed(&self) -> Duration {
        let last = self.last_activity_millis.load(Ordering::Relaxed);
        let now = Self::current_millis();
        Duration::from_millis(now.saturating_sub(last))
    }

    /// Returns true if this tracker has been inactive for longer than the threshold
    pub fn is_stale(&self, threshold: Duration) -> bool {
        self.last_activity_elapsed() > threshold
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

    /// Local actors indexed by local_id (O(1) lookup by ActorId)
    local_actors: Arc<DashMap<u64, LocalActorHandle>>,

    /// Actor name to local_id mapping (for name-based lookups)
    actor_names: Arc<DashMap<String, u64>>,

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
        let local_actors: Arc<DashMap<u64, LocalActorHandle>> = Arc::new(DashMap::new());
        let actor_names: Arc<DashMap<String, u64>> = Arc::new(DashMap::new());
        let named_actor_paths: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
        let cluster_holder: Arc<RwLock<Option<Arc<dyn NamingBackend>>>> =
            Arc::new(RwLock::new(None));
        let lifecycle = Arc::new(ActorLifecycle::new());

        // Create message handler (needs cluster reference for gossip)
        let handler = SystemMessageHandler::new(
            node_id,
            local_actors.clone(),
            actor_names.clone(),
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
            actor_names: actor_names.clone(),
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
            .start_system_actor(actor_names, named_actor_paths)
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
        actor_names: Arc<DashMap<String, u64>>,
        named_actor_paths: Arc<DashMap<String, String>>,
    ) -> anyhow::Result<()> {
        // Create SystemRef for SystemActor
        // Note: SystemRef uses a simplified DashMap<String, Sender> for sending messages
        let local_actors_ref = self.local_actors.clone();

        // Build a name -> sender mapping for SystemRef
        let local_actor_senders: Arc<DashMap<String, mpsc::Sender<Envelope>>> =
            Arc::new(DashMap::new());
        for entry in actor_names.iter() {
            let name = entry.key().clone();
            let local_id = *entry.value();
            if let Some(handle) = local_actors_ref.get(&local_id) {
                local_actor_senders.insert(name, handle.sender.clone());
            }
        }

        let system_ref = Arc::new(SystemRef {
            node_id: self.node_id,
            addr: self.addr,
            local_actors: local_actor_senders,
            named_actor_paths,
        });

        // Create SystemActor with default factory
        let system_actor = SystemActor::with_default_factory(system_ref);

        // Spawn as named actor with path "system" (use new_system to bypass namespace check)
        let system_path = ActorPath::new_system(SYSTEM_ACTOR_PATH)?;
        self.spawn_named(system_path, system_actor).await?;

        // Note: The local_actors_ref and actor_names_ref are used internally,
        // SystemRef snapshot may become stale for new actors but that's acceptable
        // since SystemActor doesn't need real-time actor list

        tracing::debug!(path = SYSTEM_ACTOR_PATH, "SystemActor started");
        Ok(())
    }

    /// Start SystemActor with custom factory (for Python extension)
    pub async fn start_system_actor_with_factory(
        self: &Arc<Self>,
        factory: BoxedActorFactory,
    ) -> anyhow::Result<()> {
        // Check if already started
        if self.actor_names.contains_key(SYSTEM_ACTOR_PATH) {
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

        // Spawn as named actor (use new_system to bypass namespace check)
        let system_path = ActorPath::new_system(SYSTEM_ACTOR_PATH)?;
        self.spawn_named(system_path, system_actor).await?;

        tracing::debug!(
            path = SYSTEM_ACTOR_PATH,
            "SystemActor started with custom factory"
        );
        Ok(())
    }

    /// Get SystemActor reference
    pub async fn system(&self) -> anyhow::Result<ActorRef> {
        self.resolve_named(&ActorPath::new_system(SYSTEM_ACTOR_PATH)?, None)
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
        self.actor_names.iter().map(|e| e.key().clone()).collect()
    }

    /// Get a local actor reference by name
    ///
    /// Returns None if the actor doesn't exist locally.
    /// This is an O(1) operation.
    pub fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef> {
        self.actor_names.get(name).and_then(|local_id| {
            self.local_actors
                .get(local_id.value())
                .map(|handle| ActorRef::local(handle.actor_id, handle.sender.clone()))
        })
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

    /// Spawn an anonymous actor (no name, only accessible via ActorRef)
    ///
    /// Note: Anonymous actors do not support supervision/restart because they have
    /// no stable identity for re-resolution. Use `spawn_named_factory` for actors
    /// that need supervision.
    pub async fn spawn_anonymous<A>(self: &Arc<Self>, actor: A) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        self.spawn_anonymous_with_options(actor.into_actor(), SpawnOptions::default())
            .await
    }

    /// Spawn an anonymous actor with custom options
    pub async fn spawn_anonymous_with_options<A>(
        self: &Arc<Self>,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        let actor = actor.into_actor();
        let actor_id = self.next_actor_id();

        // Use configured mailbox capacity
        let capacity = options
            .mailbox_capacity
            .unwrap_or(self.default_mailbox_capacity);
        let mailbox = Mailbox::with_capacity(capacity);
        let (sender, receiver) = mailbox.split();

        let stats = Arc::new(ActorStats::default());

        // Create a child cancellation token for this specific actor
        // When system shuts down, parent token cancels all children
        // When stopping individual actor, only this child token is cancelled
        let actor_cancel = self.cancel_token.child_token();

        // Create context with system reference
        let ctx = ActorContext::with_system(
            actor_id,
            self.clone() as Arc<dyn ActorSystemRef>,
            actor_cancel.clone(),
            sender.clone(),
        );

        // Spawn actor loop (no supervision for anonymous actors, they can't restart without a factory)
        let stats_clone = stats.clone();
        let cancel = actor_cancel.clone();
        let actor_id_for_log = actor_id;

        let join_handle = tokio::spawn(async move {
            let mut receiver = receiver;
            let mut ctx = ctx;
            let reason =
                run_actor_instance(actor, &mut receiver, &mut ctx, cancel, stats_clone).await;
            tracing::debug!(actor_id = ?actor_id_for_log, reason = ?reason, "Anonymous actor stopped");
        });

        // Register using local_id as key (O(1) lookup by ActorId)
        let local_id = actor_id.local_id();
        let handle = LocalActorHandle {
            sender: sender.clone(),
            join_handle,
            cancel_token: actor_cancel,
            stats: stats.clone(),
            metadata: options.metadata.clone(),
            named_path: None,
            actor_id,
        };

        // Use local_id as primary key
        self.local_actors.insert(local_id, handle);
        // Anonymous actors use their local_id as "name" for internal tracking
        self.actor_names.insert(actor_id.to_string(), local_id);

        Ok(ActorRef::local(actor_id, sender))
    }

    /// Spawn a named actor (resolvable by name across the cluster)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Name is used as both path (for resolution) and local name
    /// system.spawn_named("services/echo", MyActor).await?;
    /// ```
    pub async fn spawn_named<P, A>(self: &Arc<Self>, name: P, actor: A) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        A: IntoActor,
    {
        let path = name.into_actor_path()?;
        self.spawn_named_factory(
            path,
            Self::once_factory(actor.into_actor()),
            SpawnOptions::default(),
        )
        .await
    }

    /// Spawn a named actor with custom options
    pub async fn spawn_named_with_options<P, A>(
        self: &Arc<Self>,
        name: P,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        A: IntoActor,
    {
        let path = name.into_actor_path()?;
        self.spawn_named_factory(path, Self::once_factory(actor.into_actor()), options)
            .await
    }

    /// Spawn a named actor using a factory function
    pub async fn spawn_named_factory<P, F, A>(
        self: &Arc<Self>,
        name: P,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor,
    {
        let path = name.into_actor_path()?;
        let name_str = path.as_str();

        // Check for duplicate name
        if self.actor_names.contains_key(&name_str.to_string()) {
            return Err(anyhow::anyhow!("Actor already exists: {}", name_str));
        }

        // Check for duplicate named path
        if self.named_actor_paths.contains_key(&name_str.to_string()) {
            return Err(anyhow::anyhow!(
                "Named path already registered: {}",
                name_str
            ));
        }

        let actor_id = self.next_actor_id();
        let local_id = actor_id.local_id();

        // Use configured mailbox capacity
        let capacity = options
            .mailbox_capacity
            .unwrap_or(self.default_mailbox_capacity);
        let mailbox = Mailbox::with_capacity(capacity);
        let (sender, receiver) = mailbox.split();

        let stats = Arc::new(ActorStats::default());
        let metadata = options.metadata.clone();

        // Create a child cancellation token for this specific actor
        let actor_cancel = self.cancel_token.child_token();

        // Create context with system reference and named path
        let ctx = ActorContext::with_system_and_name(
            actor_id,
            self.clone() as Arc<dyn ActorSystemRef>,
            actor_cancel.clone(),
            sender.clone(),
            Some(name_str.to_string()),
        );

        // Spawn actor loop
        let stats_clone = stats.clone();
        let cancel = actor_cancel.clone();
        let actor_id_for_log = actor_id;
        let supervision = options.supervision.clone();

        let join_handle = tokio::spawn(async move {
            let reason =
                run_supervision_loop(factory, receiver, ctx, cancel, stats_clone, supervision)
                    .await;
            tracing::debug!(actor_id = ?actor_id_for_log, reason = ?reason, "Actor stopped");
        });

        // Register actor using local_id as primary key
        let handle = LocalActorHandle {
            sender: sender.clone(),
            join_handle,
            cancel_token: actor_cancel,
            stats: stats.clone(),
            metadata: metadata.clone(),
            named_path: Some(path.clone()),
            actor_id,
        };

        self.local_actors.insert(local_id, handle);
        self.actor_names.insert(name_str.to_string(), local_id);
        self.named_actor_paths
            .insert(name_str.to_string(), name_str.to_string());

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
    ///
    /// This is an O(1) operation for local actors using local_id indexing.
    pub async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef> {
        // Check if local
        if id.node() == self.node_id || id.node().is_local() {
            // O(1) lookup by local_id
            let handle = self
                .local_actors
                .get(&id.local_id())
                .ok_or_else(|| anyhow::anyhow!("Local actor not found: {}", id))?;
            return Ok(ActorRef::local(handle.actor_id, handle.sender.clone()));
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

            // Look up local_id from actor_names, then get handle
            let local_id = self
                .actor_names
                .get(&actor_name)
                .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", actor_name))?;

            let handle = self
                .local_actors
                .get(local_id.value())
                .ok_or_else(|| anyhow::anyhow!("Actor handle not found: {}", actor_name))?;

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

    /// Clean up stale node load trackers to prevent memory leaks
    ///
    /// Removes entries for nodes that have not been active for longer than the threshold.
    /// Call this periodically (e.g., every few minutes) in long-running systems.
    ///
    /// # Arguments
    /// * `stale_threshold` - Remove trackers inactive for longer than this duration
    ///
    /// # Returns
    /// Number of entries removed
    pub fn cleanup_stale_node_trackers(&self, stale_threshold: Duration) -> usize {
        let before = self.node_load.len();
        self.node_load.retain(|_addr, tracker| {
            // Keep entries that are still active OR have in-flight requests
            !tracker.is_stale(stale_threshold) || tracker.load() > 0
        });
        let removed = before - self.node_load.len();
        if removed > 0 {
            tracing::debug!(
                removed = removed,
                remaining = self.node_load.len(),
                "Cleaned up stale node load trackers"
            );
        }
        removed
    }

    /// Get the number of tracked nodes
    pub fn tracked_node_count(&self) -> usize {
        self.node_load.len()
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

    /// Resolve all instances of a named actor as ActorRefs
    pub async fn resolve_all_instances(
        &self,
        path: &ActorPath,
        filter_alive: bool,
    ) -> anyhow::Result<Vec<ActorRef>> {
        let cluster_guard = self.cluster.read().await;
        let cluster = cluster_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cluster not initialized"))?;

        let instances = cluster.get_named_actor_instances_detailed(path).await;

        let mut refs = Vec::new();
        for (member, instance_opt) in instances {
            // Filter by alive status if requested
            if filter_alive && member.status != MemberStatus::Alive {
                continue;
            }

            // Get actor_id from instance info
            if let Some(instance) = instance_opt {
                let actor_ref = self.actor_ref(&instance.actor_id).await?;
                refs.push(actor_ref);
            }
        }

        Ok(refs)
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

    /// Default timeout for graceful actor shutdown (30 seconds)
    const GRACEFUL_STOP_TIMEOUT: Duration = Duration::from_secs(30);

    /// Stop an actor gracefully
    ///
    /// This method first signals the actor to stop via its cancellation token,
    /// waits for it to finish (with timeout), then performs cleanup.
    /// If the actor doesn't stop within the timeout, it will be forcefully aborted.
    pub async fn stop(&self, name: impl AsRef<str>) -> anyhow::Result<()> {
        self.stop_with_reason(name, StopReason::Killed).await
    }

    /// Stop an actor with a specific reason
    ///
    /// Performs graceful shutdown:
    /// 1. Cancels the actor's cancellation token (triggers `on_stop()`)
    /// 2. Waits for the actor to finish (with 30s timeout)
    /// 3. If timeout, forcefully aborts the actor task
    /// 4. Handles lifecycle cleanup (watch notifications, cluster broadcast, etc.)
    ///
    /// Note: If the name doesn't contain a "/" and no actor is found with the exact name,
    /// it will try with the "actors/" prefix (for Python compatibility).
    pub async fn stop_with_reason(
        &self,
        name: impl AsRef<str>,
        reason: StopReason,
    ) -> anyhow::Result<()> {
        let name = name.as_ref();

        // Try exact name first, then normalized name with "actors/" prefix
        let actual_name = if self.actor_names.contains_key(name) {
            name.to_string()
        } else if !name.contains('/') {
            // Try with "actors/" prefix (Python API compatibility)
            let prefixed = format!("actors/{}", name);
            if self.actor_names.contains_key(&prefixed) {
                prefixed
            } else {
                name.to_string()
            }
        } else {
            name.to_string()
        };

        // Get local_id from actor_names, then remove from local_actors
        if let Some((_, local_id)) = self.actor_names.remove(&actual_name) {
            if let Some((_, handle)) = self.local_actors.remove(&local_id) {
                // 1. Signal the actor to stop gracefully
                handle.cancel_token.cancel();

                // 2. Wait for the actor to finish with timeout
                match tokio::time::timeout(Self::GRACEFUL_STOP_TIMEOUT, handle.join_handle).await {
                    Ok(_) => {
                        // Actor stopped gracefully
                        tracing::debug!(actor = %actual_name, "Actor stopped gracefully");
                    }
                    Err(_) => {
                        // Timeout - actor didn't respond to cancel signal
                        // This shouldn't happen normally, but we log a warning
                        tracing::warn!(
                            actor = %actual_name,
                            "Actor didn't stop gracefully within timeout, already aborted by tokio"
                        );
                    }
                }

                // 3. Handle lifecycle cleanup
                let actor_names = self.actor_names.clone();
                let local_actors = self.local_actors.clone();
                self.lifecycle
                    .handle_termination(
                        &handle.actor_id,
                        &actual_name,
                        handle.named_path.clone(),
                        reason,
                        &self.named_actor_paths,
                        &self.cluster,
                        |n| {
                            actor_names.get(n).and_then(|id| {
                                local_actors.get(id.value()).map(|h| h.sender.clone())
                            })
                        },
                    )
                    .await;
            }
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

            // Get local_id from actor_names, then remove from local_actors
            if let Some((_, local_id)) = self.actor_names.remove(&actor_name) {
                if let Some((_, handle)) = self.local_actors.remove(&local_id) {
                    // 1. Signal the actor to stop gracefully
                    handle.cancel_token.cancel();

                    // 2. Wait for the actor to finish with timeout
                    match tokio::time::timeout(Self::GRACEFUL_STOP_TIMEOUT, handle.join_handle)
                        .await
                    {
                        Ok(_) => {
                            tracing::debug!(actor = %actor_name, path = %path_key, "Actor stopped gracefully");
                        }
                        Err(_) => {
                            tracing::warn!(
                                actor = %actor_name,
                                path = %path_key,
                                "Actor didn't stop gracefully within timeout"
                            );
                        }
                    }

                    // 3. Handle lifecycle cleanup
                    let actor_names = self.actor_names.clone();
                    let local_actors = self.local_actors.clone();
                    self.lifecycle
                        .handle_termination(
                            &handle.actor_id,
                            &actor_name,
                            Some(path.clone()),
                            reason,
                            &self.named_actor_paths,
                            &self.cluster,
                            |name| {
                                actor_names.get(name).and_then(|id| {
                                    local_actors.get(id.value()).map(|h| h.sender.clone())
                                })
                            },
                        )
                        .await;
                }
            }
        }

        Ok(())
    }

    /// Shutdown the entire actor system
    ///
    /// This method performs a graceful shutdown:
    /// 1. Signals cancellation to all actors (via parent cancel token, which cancels all child tokens)
    /// 2. Waits for actors to stop gracefully (with timeout)
    /// 3. Triggers lifecycle cleanup for each actor (watch notifications, cluster broadcast, etc.)
    /// 4. Leaves the cluster gracefully
    /// 5. Clears all actors and watch relationships
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        tracing::info!("Shutting down actor system");

        // Signal cancellation first - this cancels the parent token,
        // which automatically cancels all child tokens (individual actor tokens)
        // This triggers the `cancel.cancelled()` branch in each actor's message loop,
        // allowing them to call `on_stop()` gracefully
        self.cancel_token.cancel();

        // Give actors a short time to process the cancellation signal
        // Since all actors share the same parent token, they should all start stopping
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Collect all actor info (local_id, actor_id, name, named_path)
        let actor_entries: Vec<_> = self
            .local_actors
            .iter()
            .map(|entry| {
                let local_id = *entry.key();
                let actor_id = entry.actor_id;
                let named_path = entry.named_path.clone();
                // Find name from actor_names (reverse lookup)
                let name = self
                    .actor_names
                    .iter()
                    .find(|e| *e.value() == local_id)
                    .map(|e| e.key().clone())
                    .unwrap_or_else(|| actor_id.to_string());
                (local_id, actor_id, name, named_path)
            })
            .collect();

        // Process each actor's termination
        for (local_id, actor_id, actor_name, named_path) in actor_entries {
            // Remove from actor_names first
            self.actor_names.remove(&actor_name);

            // Remove and get ownership of the handle
            if let Some((_, handle)) = self.local_actors.remove(&local_id) {
                // Wait briefly for graceful shutdown (actor should already be stopping due to parent cancel)
                // Use a shorter timeout since we already signaled cancellation
                match tokio::time::timeout(Duration::from_secs(5), handle.join_handle).await {
                    Ok(_) => {
                        tracing::debug!(actor = %actor_name, "Actor stopped gracefully during shutdown");
                    }
                    Err(_) => {
                        // Timeout - this shouldn't happen normally since cancel was already called
                        tracing::warn!(
                            actor = %actor_name,
                            "Actor didn't stop within timeout during shutdown"
                        );
                    }
                }

                // Trigger lifecycle cleanup (watch notifications, cluster broadcast, routing cleanup)
                let actor_names = self.actor_names.clone();
                let local_actors = self.local_actors.clone();
                self.lifecycle
                    .handle_termination(
                        &actor_id,
                        &actor_name,
                        named_path,
                        StopReason::SystemShutdown,
                        &self.named_actor_paths,
                        &self.cluster,
                        |name| {
                            actor_names.get(name).and_then(|id| {
                                local_actors.get(id.value()).map(|h| h.sender.clone())
                            })
                        },
                    )
                    .await;
            }
        }

        // Clear all actors (should already be empty, but just in case)
        self.local_actors.clear();
        self.actor_names.clear();

        // Clear node load trackers
        self.node_load.clear();

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

    fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef> {
        ActorSystem::local_actor_ref_by_name(self, name)
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
