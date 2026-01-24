//! Actor System Extension Traits
//!
//! This module defines the public API surface for ActorSystem through traits:
//! - [`ActorSystemCoreExt`] - Core spawn and resolve operations (primary API)
//! - [`ActorSystemAdvancedExt`] - Factory-based spawning for supervision/restart
//! - [`ActorSystemOpsExt`] - Operations, introspection, and lifecycle management

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::actor::{Actor, ActorId, ActorPath, ActorRef, IntoActor, IntoActorPath, NodeId};
use crate::cluster::{MemberInfo, NamedActorInfo};
use crate::supervision::SupervisionSpec;
use crate::system_actor::BoxedActorFactory;

use super::config::{ResolveOptions, SpawnOptions};
use super::NodeLoadTracker;
use crate::policies::LoadBalancingPolicy;

use tokio_util::sync::CancellationToken;

// =============================================================================
// Core Trait: Spawn + Resolve (Primary API)
// =============================================================================

/// Core API for spawning and resolving actors.
///
/// This trait defines the primary interface for creating and locating actors.
/// It is automatically implemented for `Arc<ActorSystem>` and re-exported in prelude.
///
/// # Spawn Methods
/// - [`spawn`](Self::spawn) - Spawn an anonymous actor (not resolvable by name)
/// - [`spawn_named`](Self::spawn_named) - Spawn a named actor (resolvable by name)
/// - [`spawning`](Self::spawning) - Get a builder for advanced spawn options
///
/// # Resolve Methods
/// - [`actor_ref`](Self::actor_ref) - Get ActorRef by ActorId
/// - [`resolve`](Self::resolve) - Resolve a named actor by name
/// - [`resolve_with_options`](Self::resolve_with_options) - Resolve with load balancing/filtering
/// - [`resolve_lazy`](Self::resolve_lazy) - Lazy resolution with auto-refresh
///
/// # Example
/// ```rust,ignore
/// use pulsing_actor::prelude::*;
///
/// let system = ActorSystem::builder().build().await?;
///
/// // Spawn an anonymous actor (only accessible via ActorRef)
/// let worker = system.spawn(Worker::new()).await?;
///
/// // Spawn a named actor (resolvable by name)
/// let echo = system.spawn_named("services/echo", EchoService).await?;
///
/// // Spawn with builder for advanced options
/// let counter = system.spawning()
///     .name("services/counter")
///     .supervision(SupervisionSpec::on_failure().max_restarts(3))
///     .mailbox_capacity(256)
///     .spawn(Counter::new())
///     .await?;
///
/// // Resolve by name
/// let echo_ref = system.resolve("services/echo").await?;
/// ```
#[async_trait::async_trait]
pub trait ActorSystemCoreExt: Sized {
    /// Spawn an anonymous actor (not resolvable by name, only accessible via ActorRef)
    ///
    /// Accepts any type that implements `IntoActor`, including:
    /// - Types implementing `Actor` directly
    /// - `Behavior<M>` (automatically wrapped)
    async fn spawn<A>(&self, actor: A) -> anyhow::Result<ActorRef>
    where
        A: IntoActor;

    /// Spawn a named actor (resolvable by name across the cluster)
    ///
    /// Named actors can be discovered and resolved by other nodes using [`resolve`](Self::resolve).
    ///
    /// Accepts any type that implements `IntoActor`, including:
    /// - Types implementing `Actor` directly
    /// - `Behavior<M>` (automatically wrapped)
    ///
    /// # Arguments
    /// - `name` - The name for discovery (e.g., "services/echo")
    /// - `actor` - The actor instance or Behavior
    async fn spawn_named<A>(
        &self,
        name: impl AsRef<str> + Send,
        actor: A,
    ) -> anyhow::Result<ActorRef>
    where
        A: IntoActor;

    /// Get a builder for spawning actors with advanced options
    ///
    /// # Example
    /// ```rust,ignore
    /// let actor = system.spawning()
    ///     .name("services/worker")
    ///     .supervision(SupervisionSpec::on_failure().max_restarts(3))
    ///     .mailbox_capacity(1024)
    ///     .spawn(Worker::new())
    ///     .await?;
    /// ```
    fn spawning(&self) -> SpawnBuilder<'_>;

    /// Get ActorRef for a local or remote actor by ID
    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef>;

    /// Resolve a named actor by name
    ///
    /// Returns an ActorRef that points to the current location of the named actor.
    /// Note: If the actor migrates, this reference may become stale.
    /// For actors that may migrate, consider using [`resolve_lazy`](Self::resolve_lazy).
    async fn resolve<P>(&self, name: P) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send;

    /// Resolve a named actor with custom options (load balancing, node filtering)
    async fn resolve_with_options(
        &self,
        name: &ActorPath,
        options: ResolveOptions,
    ) -> anyhow::Result<ActorRef>;

    /// Get a builder for resolving actors with advanced options
    ///
    /// # Example
    /// ```rust,ignore
    /// // With load balancing
    /// let actor = system.resolving()
    ///     .policy(RoundRobinPolicy::new())
    ///     .resolve("services/worker").await?;
    ///
    /// // List all instances
    /// let actors = system.resolving()
    ///     .list("services/worker").await?;
    ///
    /// // Lazy resolve
    /// let actor = system.resolving()
    ///     .lazy("services/worker")?;
    /// ```
    fn resolving(&self) -> ResolveBuilder<'_>;
}

// =============================================================================
// SpawnBuilder: Fluent API for spawning actors
// =============================================================================

/// Builder for spawning actors with advanced options.
///
/// # Example
/// ```rust,ignore
/// // Anonymous actor with supervision
/// let worker = system.spawning()
///     .supervision(SupervisionSpec::on_failure().max_restarts(3))
///     .spawn(Worker::new())
///     .await?;
///
/// // Named actor with full options
/// let service = system.spawning()
///     .name("services/counter")
///     .supervision(SupervisionSpec::on_failure().max_restarts(5))
///     .mailbox_capacity(512)
///     .spawn(CounterService::new())
///     .await?;
/// ```
pub struct SpawnBuilder<'a> {
    system: &'a Arc<ActorSystem>,
    name: Option<String>,
    options: SpawnOptions,
}

impl<'a> SpawnBuilder<'a> {
    /// Create a new SpawnBuilder
    pub(crate) fn new(system: &'a Arc<ActorSystem>) -> Self {
        Self {
            system,
            name: None,
            options: SpawnOptions::default(),
        }
    }

    /// Set the actor name (makes it resolvable by name)
    pub fn name(mut self, name: impl AsRef<str>) -> Self {
        self.name = Some(name.as_ref().to_string());
        self
    }

    /// Set supervision specification (restart policy)
    pub fn supervision(mut self, spec: SupervisionSpec) -> Self {
        self.options.supervision = spec;
        self
    }

    /// Set mailbox capacity
    pub fn mailbox_capacity(mut self, capacity: usize) -> Self {
        self.options.mailbox_capacity = Some(capacity);
        self
    }

    /// Set actor metadata
    pub fn metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.options.metadata = metadata;
        self
    }

    /// Spawn the actor
    ///
    /// Accepts any type that implements `IntoActor`, including:
    /// - Types implementing `Actor` directly
    /// - `Behavior<M>` (automatically wrapped)
    ///
    /// If a name was set, spawns a named actor (resolvable).
    /// Otherwise, spawns an anonymous actor (only accessible via ActorRef).
    pub async fn spawn<A>(self, actor: A) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        let actor = actor.into_actor();
        match self.name {
            Some(name) => {
                // Named actor: resolvable by name
                ActorSystem::spawn_named_with_options(
                    self.system,
                    name.as_str(),
                    actor,
                    self.options,
                )
                .await
            }
            None => {
                // Anonymous actor: not resolvable
                ActorSystem::spawn_anonymous_with_options(self.system, actor, self.options).await
            }
        }
    }
}

// =============================================================================
// ResolveBuilder: Fluent API for resolving actors
// =============================================================================

/// Builder for resolving actors with advanced options.
///
/// # Example
/// ```rust,ignore
/// // Simple resolve
/// let actor = system.resolve("services/counter").await?;
///
/// // With load balancing policy
/// let actor = system.resolving()
///     .policy(RoundRobinPolicy::new())
///     .resolve("services/counter").await?;
///
/// // Get all instances
/// let actors = system.resolving()
///     .list("services/counter").await?;
///
/// // Lazy resolve (auto re-resolves on stale)
/// let actor = system.resolving()
///     .lazy("services/counter")?;
/// ```
pub struct ResolveBuilder<'a> {
    system: &'a Arc<ActorSystem>,
    node_id: Option<NodeId>,
    policy: Option<Arc<dyn LoadBalancingPolicy>>,
    filter_alive: bool,
}

impl<'a> ResolveBuilder<'a> {
    /// Create a new ResolveBuilder
    pub(crate) fn new(system: &'a Arc<ActorSystem>) -> Self {
        Self {
            system,
            node_id: None,
            policy: None,
            filter_alive: true,
        }
    }

    /// Target a specific node (bypasses load balancing)
    pub fn node(mut self, node_id: NodeId) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Set load balancing policy
    pub fn policy(mut self, policy: Arc<dyn LoadBalancingPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Set whether to filter only alive nodes (default: true)
    pub fn filter_alive(mut self, filter: bool) -> Self {
        self.filter_alive = filter;
        self
    }

    /// Build ResolveOptions from this builder
    fn build_options(&self) -> ResolveOptions {
        let mut options = ResolveOptions::new();
        if let Some(node_id) = self.node_id {
            options = options.node_id(node_id);
        }
        if let Some(ref policy) = self.policy {
            options = options.policy(policy.clone());
        }
        options = options.filter_alive(self.filter_alive);
        options
    }

    /// Resolve a named actor
    pub async fn resolve<P>(self, name: P) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
    {
        let path = name.into_actor_path()?;
        let options = self.build_options();
        ActorSystem::resolve_named_with_options(self.system, &path, options).await
    }

    /// List all instances of a named actor
    pub async fn list<P>(self, name: P) -> anyhow::Result<Vec<ActorRef>>
    where
        P: IntoActorPath + Send,
    {
        let path = name.into_actor_path()?;
        ActorSystem::resolve_all_instances(self.system, &path, self.filter_alive).await
    }

    /// Lazy resolve - returns ActorRef that auto re-resolves when stale
    pub fn lazy<P>(self, name: P) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
    {
        ActorSystem::resolve_named_lazy(self.system, name)
    }
}

// =============================================================================
// Advanced Trait: Factory-based Spawning (Supervision/Restart)
// =============================================================================

/// Advanced API for factory-based actor spawning.
///
/// Factory-based spawning enables supervision restarts - when an actor fails,
/// the system can recreate it using the factory function.
///
/// Note: Regular `spawn` methods use a one-shot factory internally, so the actor
/// cannot be restarted. Use `spawn_named_factory` if you need supervision with
/// restart capability. Anonymous actors do not support supervision.
///
/// # Example
/// ```rust,ignore
/// use pulsing_actor::prelude::*;
///
/// let system = ActorSystem::builder().build().await?;
///
/// let options = SpawnOptions::new()
///     .supervision(SupervisionSpec::new()
///         .restart_policy(RestartPolicy::OnFailure)
///         .max_restarts(3));
///
/// // Spawn named actor with factory (only named actors support supervision)
/// let named = system.spawn_named_factory("services/worker", || Ok(Worker::new()), options).await?;
/// ```
#[async_trait::async_trait]
pub trait ActorSystemAdvancedExt {
    /// Spawn a named actor using a factory function (enables supervision restarts)
    ///
    /// Note: Only named actors support supervision/restart. Anonymous actors cannot
    /// be restarted because they have no stable identity for re-resolution.
    async fn spawn_named_factory<P, F, A>(
        &self,
        name: P,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor;
}

// =============================================================================
// Ops Trait: Operations, Introspection, Lifecycle
// =============================================================================

/// Operations, introspection, and lifecycle management API.
///
/// This trait provides:
/// - System information (node_id, addr, etc.)
/// - Actor listing and lookup
/// - Cluster membership information
/// - Actor stop and system shutdown
///
/// # Example
/// ```rust,ignore
/// use pulsing_actor::prelude::*;
///
/// let system = ActorSystem::builder().build().await?;
///
/// // Get system info
/// println!("Node ID: {}", system.node_id());
/// println!("Address: {}", system.addr());
///
/// // List cluster members
/// for member in system.members().await {
///     println!("Member: {} at {}", member.node_id, member.addr);
/// }
///
/// // Shutdown
/// system.shutdown().await?;
/// ```
#[async_trait::async_trait]
pub trait ActorSystemOpsExt {
    /// Get SystemActor reference
    async fn system(&self) -> anyhow::Result<ActorRef>;

    /// Start SystemActor with custom factory (for Python extension)
    async fn start_system_actor_with_factory(
        &self,
        factory: BoxedActorFactory,
    ) -> anyhow::Result<()>;

    /// Get node ID
    fn node_id(&self) -> &NodeId;

    /// Get local address
    fn addr(&self) -> SocketAddr;

    /// Get list of local actor names
    fn local_actor_names(&self) -> Vec<String>;

    /// Get a local actor reference by name
    fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef>;

    /// Spawn an anonymous actor (no name, only accessible via ActorRef)
    async fn spawn_anonymous<A>(&self, actor: A) -> anyhow::Result<ActorRef>
    where
        A: IntoActor;

    /// Spawn an anonymous actor with custom options
    async fn spawn_anonymous_with_options<A>(
        &self,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: IntoActor;

    /// Get load tracker for a node address
    fn get_node_load_tracker(&self, addr: &SocketAddr) -> Option<Arc<NodeLoadTracker>>;

    /// Decrement load after a request completes
    fn decrement_node_load(&self, addr: &SocketAddr);

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
    fn cleanup_stale_node_trackers(&self, stale_threshold: std::time::Duration) -> usize;

    /// Get the number of tracked nodes
    fn tracked_node_count(&self) -> usize;

    /// Resolve an actor address and get an ActorRef
    async fn resolve_address(
        &self,
        address: &crate::actor::ActorAddress,
    ) -> anyhow::Result<ActorRef>;

    /// Get all instances of a named actor across the cluster
    async fn get_named_instances(&self, path: &ActorPath) -> Vec<MemberInfo>;

    /// Get detailed instances with actor_id and metadata
    async fn get_named_instances_detailed(
        &self,
        path: &ActorPath,
    ) -> Vec<(MemberInfo, Option<crate::cluster::NamedActorInstance>)>;

    /// Get all named actors in the cluster
    async fn all_named_actors(&self) -> Vec<NamedActorInfo>;

    /// Lookup named actor information
    async fn lookup_named(&self, path: &ActorPath) -> Option<NamedActorInfo>;

    /// Get cluster member information
    async fn members(&self) -> Vec<MemberInfo>;

    /// Stop an actor by local name
    async fn stop(&self, name: impl AsRef<str> + Send) -> anyhow::Result<()>;

    /// Stop an actor with a specific reason
    async fn stop_with_reason(
        &self,
        name: impl AsRef<str> + Send,
        reason: crate::actor::StopReason,
    ) -> anyhow::Result<()>;

    /// Stop a named actor by path
    async fn stop_named(&self, path: &ActorPath) -> anyhow::Result<()>;

    /// Stop a named actor by path with a specific reason
    async fn stop_named_with_reason(
        &self,
        path: &ActorPath,
        reason: crate::actor::StopReason,
    ) -> anyhow::Result<()>;

    /// Shutdown the entire actor system
    async fn shutdown(&self) -> anyhow::Result<()>;

    /// Get cancellation token
    fn cancel_token(&self) -> CancellationToken;
}

// =============================================================================
// Implementations for Arc<ActorSystem>
// =============================================================================

use super::ActorSystem;

#[async_trait::async_trait]
impl ActorSystemCoreExt for Arc<ActorSystem> {
    async fn spawn<A>(&self, actor: A) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        ActorSystem::spawn_anonymous(self, actor.into_actor()).await
    }

    async fn spawn_named<A>(
        &self,
        name: impl AsRef<str> + Send,
        actor: A,
    ) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        let name = name.as_ref();
        ActorSystem::spawn_named_with_options(
            self,
            name,
            actor.into_actor(),
            SpawnOptions::default(),
        )
        .await
    }

    fn spawning(&self) -> SpawnBuilder<'_> {
        SpawnBuilder::new(self)
    }

    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef> {
        ActorSystem::actor_ref(self.as_ref(), id).await
    }

    async fn resolve<P>(&self, name: P) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
    {
        ActorSystem::resolve_named(self.as_ref(), name, None).await
    }

    async fn resolve_with_options(
        &self,
        name: &ActorPath,
        options: ResolveOptions,
    ) -> anyhow::Result<ActorRef> {
        ActorSystem::resolve_named_with_options(self.as_ref(), name, options).await
    }

    fn resolving(&self) -> ResolveBuilder<'_> {
        ResolveBuilder::new(self)
    }
}

#[async_trait::async_trait]
impl ActorSystemAdvancedExt for Arc<ActorSystem> {
    async fn spawn_named_factory<P, F, A>(
        &self,
        name: P,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor,
    {
        ActorSystem::spawn_named_factory(self, name, factory, options).await
    }
}

#[async_trait::async_trait]
impl ActorSystemOpsExt for Arc<ActorSystem> {
    async fn system(&self) -> anyhow::Result<ActorRef> {
        ActorSystem::system(self.as_ref()).await
    }

    async fn start_system_actor_with_factory(
        &self,
        factory: BoxedActorFactory,
    ) -> anyhow::Result<()> {
        ActorSystem::start_system_actor_with_factory(self, factory).await
    }

    fn node_id(&self) -> &NodeId {
        ActorSystem::node_id(self.as_ref())
    }

    fn addr(&self) -> SocketAddr {
        ActorSystem::addr(self.as_ref())
    }

    fn local_actor_names(&self) -> Vec<String> {
        ActorSystem::local_actor_names(self.as_ref())
    }

    fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef> {
        ActorSystem::local_actor_ref_by_name(self.as_ref(), name)
    }

    async fn spawn_anonymous<A>(&self, actor: A) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        ActorSystem::spawn_anonymous(self, actor).await
    }

    async fn spawn_anonymous_with_options<A>(
        &self,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: IntoActor,
    {
        ActorSystem::spawn_anonymous_with_options(self, actor, options).await
    }

    fn get_node_load_tracker(&self, addr: &SocketAddr) -> Option<Arc<NodeLoadTracker>> {
        ActorSystem::get_node_load_tracker(self.as_ref(), addr)
    }

    fn decrement_node_load(&self, addr: &SocketAddr) {
        ActorSystem::decrement_node_load(self.as_ref(), addr)
    }

    fn cleanup_stale_node_trackers(&self, stale_threshold: std::time::Duration) -> usize {
        ActorSystem::cleanup_stale_node_trackers(self.as_ref(), stale_threshold)
    }

    fn tracked_node_count(&self) -> usize {
        ActorSystem::tracked_node_count(self.as_ref())
    }

    async fn resolve_address(
        &self,
        address: &crate::actor::ActorAddress,
    ) -> anyhow::Result<ActorRef> {
        ActorSystem::resolve(self.as_ref(), address).await
    }

    async fn get_named_instances(&self, path: &ActorPath) -> Vec<MemberInfo> {
        ActorSystem::get_named_instances(self.as_ref(), path).await
    }

    async fn get_named_instances_detailed(
        &self,
        path: &ActorPath,
    ) -> Vec<(MemberInfo, Option<crate::cluster::NamedActorInstance>)> {
        ActorSystem::get_named_instances_detailed(self.as_ref(), path).await
    }

    async fn all_named_actors(&self) -> Vec<NamedActorInfo> {
        ActorSystem::all_named_actors(self.as_ref()).await
    }

    async fn lookup_named(&self, path: &ActorPath) -> Option<NamedActorInfo> {
        ActorSystem::lookup_named(self.as_ref(), path).await
    }

    async fn members(&self) -> Vec<MemberInfo> {
        ActorSystem::members(self.as_ref()).await
    }

    async fn stop(&self, name: impl AsRef<str> + Send) -> anyhow::Result<()> {
        ActorSystem::stop(self.as_ref(), name).await
    }

    async fn stop_with_reason(
        &self,
        name: impl AsRef<str> + Send,
        reason: crate::actor::StopReason,
    ) -> anyhow::Result<()> {
        ActorSystem::stop_with_reason(self.as_ref(), name, reason).await
    }

    async fn stop_named(&self, path: &ActorPath) -> anyhow::Result<()> {
        ActorSystem::stop_named(self.as_ref(), path).await
    }

    async fn stop_named_with_reason(
        &self,
        path: &ActorPath,
        reason: crate::actor::StopReason,
    ) -> anyhow::Result<()> {
        ActorSystem::stop_named_with_reason(self.as_ref(), path, reason).await
    }

    async fn shutdown(&self) -> anyhow::Result<()> {
        ActorSystem::shutdown(self.as_ref()).await
    }

    fn cancel_token(&self) -> CancellationToken {
        ActorSystem::cancel_token(self.as_ref())
    }
}
