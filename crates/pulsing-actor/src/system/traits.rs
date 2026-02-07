//! ActorSystem extension traits.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::actor::{Actor, ActorId, ActorPath, ActorRef, IntoActor, IntoActorPath, NodeId};
use crate::cluster::{MemberInfo, NamedActorInfo};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::supervision::SupervisionSpec;
use crate::system_actor::BoxedActorFactory;

use super::config::{ResolveOptions, SpawnOptions};
use super::NodeLoadTracker;
use crate::policies::LoadBalancingPolicy;

use tokio_util::sync::CancellationToken;

/// Core API for spawning and resolving actors.
#[async_trait::async_trait]
pub trait ActorSystemCoreExt: Sized {
    /// Spawn an anonymous actor (not resolvable by name, only accessible via ActorRef)
    ///
    /// Accepts any type that implements `IntoActor`, including:
    /// - Types implementing `Actor` directly
    /// - `Behavior<M>` (automatically wrapped)
    async fn spawn<A>(&self, actor: A) -> Result<ActorRef>
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
    async fn spawn_named<A>(&self, name: impl AsRef<str> + Send, actor: A) -> Result<ActorRef>
    where
        A: IntoActor;

    /// Get a builder for spawning actors with advanced options.
    fn spawning(&self) -> SpawnBuilder<'_>;

    /// Get ActorRef for a local or remote actor by ID
    async fn actor_ref(&self, id: &ActorId) -> Result<ActorRef>;

    /// Resolve a named actor by name.
    async fn resolve<P>(&self, name: P) -> Result<ActorRef>
    where
        P: IntoActorPath + Send;

    /// Get a builder for resolving actors with advanced options.
    fn resolving(&self) -> ResolveBuilder<'_>;
}

/// Builder for spawning actors with advanced options.
pub struct SpawnBuilder<'a> {
    system: &'a Arc<ActorSystem>,
    name: Option<ActorPath>,
    name_error: Option<String>,
    options: SpawnOptions,
}

impl<'a> SpawnBuilder<'a> {
    /// Create a new SpawnBuilder
    pub(crate) fn new(system: &'a Arc<ActorSystem>) -> Self {
        Self {
            system,
            name: None,
            name_error: None,
            options: SpawnOptions::default(),
        }
    }

    /// Set the actor name (makes it resolvable by name)
    ///
    /// The name will be validated as an ActorPath. For user actors,
    /// use paths like "services/echo" or "actors/counter".
    ///
    /// If validation fails, the error will be stored and returned when `spawn()` or `spawn_factory()` is called.
    pub fn name(mut self, name: impl AsRef<str>) -> Self {
        match ActorPath::new(name.as_ref()) {
            Ok(path) => {
                self.name = Some(path);
                self.name_error = None; // Clear any previous error
            }
            Err(e) => {
                // Store error message for later reporting
                self.name_error = Some(format!("Invalid actor path '{}': {}", name.as_ref(), e));
                self.name = None;
                tracing::warn!("{}", self.name_error.as_ref().unwrap());
            }
        }
        self
    }

    /// Set the actor path directly (allows system paths)
    ///
    /// This method allows setting an already-validated ActorPath directly,
    /// bypassing the string validation in `name()`. This is useful when
    /// you already have an ActorPath or need to use system namespace paths.
    pub fn path(mut self, path: ActorPath) -> Self {
        self.name = Some(path);
        self.name_error = None; // Clear any previous error
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
    pub async fn spawn<A>(self, actor: A) -> Result<ActorRef>
    where
        A: IntoActor,
    {
        let actor = actor.into_actor();
        // Create a once-use factory from the actor instance
        let mut actor_opt = Some(actor);
        let factory = move || {
            actor_opt.take().ok_or_else(|| {
                PulsingError::from(RuntimeError::actor_spawn_failed(
                    "Actor cannot be restarted (spawned as instance)",
                ))
            })
        };
        self.spawn_factory(factory).await
    }

    /// Spawn an actor using a factory function
    ///
    /// Factory-based spawning enables supervision restarts - when an actor fails,
    /// the system can recreate it using the factory function.
    ///
    /// Note: Only named actors support supervision/restart. Anonymous actors
    /// cannot be restarted because they have no stable identity for re-resolution.
    pub async fn spawn_factory<F, A>(self, factory: F) -> Result<ActorRef>
    where
        F: FnMut() -> Result<A> + Send + 'static,
        A: Actor,
    {
        // Check if name validation failed
        if let Some(ref error) = self.name_error {
            return Err(PulsingError::from(RuntimeError::invalid_actor_path(
                error.clone(),
            )));
        }

        match self.name {
            Some(path) => {
                // Named actor: resolvable by name
                ActorSystem::spawn_internal(self.system, Some(path), factory, self.options).await
            }
            None => {
                // Anonymous actor: not resolvable
                ActorSystem::spawn_internal(self.system, None, factory, self.options).await
            }
        }
    }
}

/// Builder for resolving actors with advanced options.
pub struct ResolveBuilder<'a> {
    system: &'a Arc<ActorSystem>,
    options: ResolveOptions,
}

impl<'a> ResolveBuilder<'a> {
    /// Create a new ResolveBuilder
    pub(crate) fn new(system: &'a Arc<ActorSystem>) -> Self {
        Self {
            system,
            options: ResolveOptions::default(),
        }
    }

    /// Target a specific node (bypasses load balancing)
    pub fn node(mut self, node_id: NodeId) -> Self {
        self.options = self.options.node_id(node_id);
        self
    }

    /// Set load balancing policy
    pub fn policy(mut self, policy: Arc<dyn LoadBalancingPolicy>) -> Self {
        self.options = self.options.policy(policy);
        self
    }

    /// Set whether to filter only alive nodes (default: true)
    pub fn filter_alive(mut self, filter: bool) -> Self {
        self.options = self.options.filter_alive(filter);
        self
    }

    /// Resolve a named actor
    pub async fn resolve<P>(self, name: P) -> Result<ActorRef>
    where
        P: IntoActorPath + Send,
    {
        let path = name.into_actor_path()?;
        ActorSystem::resolve_named_with_options(self.system, &path, self.options).await
    }

    /// List all instances of a named actor
    pub async fn list<P>(self, name: P) -> Result<Vec<ActorRef>>
    where
        P: IntoActorPath + Send,
    {
        let path = name.into_actor_path()?;
        ActorSystem::resolve_all_instances(self.system, &path, self.options.filter_alive).await
    }

    /// Lazy resolve - returns ActorRef that auto re-resolves when stale
    pub fn lazy<P>(self, name: P) -> Result<ActorRef>
    where
        P: IntoActorPath,
    {
        ActorSystem::resolve_named_lazy(self.system, name)
    }
}

/// Operations, introspection, and lifecycle management API.
#[async_trait::async_trait]
pub trait ActorSystemOpsExt {
    /// Get SystemActor reference
    async fn system(&self) -> Result<ActorRef>;

    /// Start SystemActor with custom factory (for Python extension)
    async fn start_system_actor_with_factory(&self, factory: BoxedActorFactory) -> Result<()>;

    /// Get node ID
    fn node_id(&self) -> &NodeId;

    /// Get local address
    fn addr(&self) -> SocketAddr;

    /// Get list of local actor names
    fn local_actor_names(&self) -> Vec<String>;

    /// Get a local actor reference by name
    fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef>;

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
    async fn resolve_address(&self, address: &crate::actor::ActorAddress) -> Result<ActorRef>;

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
    async fn stop(&self, name: impl AsRef<str> + Send) -> Result<()>;

    /// Stop an actor with a specific reason
    async fn stop_with_reason(
        &self,
        name: impl AsRef<str> + Send,
        reason: crate::actor::StopReason,
    ) -> Result<()>;

    /// Stop a named actor by path
    async fn stop_named(&self, path: &ActorPath) -> Result<()>;

    /// Stop a named actor by path with a specific reason
    async fn stop_named_with_reason(
        &self,
        path: &ActorPath,
        reason: crate::actor::StopReason,
    ) -> Result<()>;

    /// Shutdown the entire actor system
    async fn shutdown(&self) -> Result<()>;

    /// Get cancellation token
    fn cancel_token(&self) -> CancellationToken;
}

// =============================================================================
// Implementations for Arc<ActorSystem>
// =============================================================================

use super::ActorSystem;

#[async_trait::async_trait]
impl ActorSystemCoreExt for Arc<ActorSystem> {
    async fn spawn<A>(&self, actor: A) -> Result<ActorRef>
    where
        A: IntoActor,
    {
        self.spawning().spawn(actor).await
    }

    async fn spawn_named<A>(&self, name: impl AsRef<str> + Send, actor: A) -> Result<ActorRef>
    where
        A: IntoActor,
    {
        self.spawning().name(name).spawn(actor).await
    }

    fn spawning(&self) -> SpawnBuilder<'_> {
        SpawnBuilder::new(self)
    }

    async fn actor_ref(&self, id: &ActorId) -> Result<ActorRef> {
        ActorSystem::actor_ref(self.as_ref(), id).await
    }

    async fn resolve<P>(&self, name: P) -> Result<ActorRef>
    where
        P: IntoActorPath + Send,
    {
        ActorSystem::resolve_named(self.as_ref(), name, None).await
    }

    fn resolving(&self) -> ResolveBuilder<'_> {
        ResolveBuilder::new(self)
    }
}

#[async_trait::async_trait]
impl ActorSystemOpsExt for Arc<ActorSystem> {
    async fn system(&self) -> Result<ActorRef> {
        ActorSystem::system(self.as_ref()).await
    }

    async fn start_system_actor_with_factory(&self, factory: BoxedActorFactory) -> Result<()> {
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

    async fn resolve_address(&self, address: &crate::actor::ActorAddress) -> Result<ActorRef> {
        ActorSystem::resolve(self.as_ref(), address).await
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

    async fn stop(&self, name: impl AsRef<str> + Send) -> Result<()> {
        ActorSystem::stop(self.as_ref(), name).await
    }

    async fn stop_with_reason(
        &self,
        name: impl AsRef<str> + Send,
        reason: crate::actor::StopReason,
    ) -> Result<()> {
        ActorSystem::stop_with_reason(self.as_ref(), name, reason).await
    }

    async fn stop_named(&self, path: &ActorPath) -> Result<()> {
        ActorSystem::stop_named(self.as_ref(), path).await
    }

    async fn stop_named_with_reason(
        &self,
        path: &ActorPath,
        reason: crate::actor::StopReason,
    ) -> Result<()> {
        ActorSystem::stop_named_with_reason(self.as_ref(), path, reason).await
    }

    async fn shutdown(&self) -> Result<()> {
        ActorSystem::shutdown(self.as_ref()).await
    }

    fn cancel_token(&self) -> CancellationToken {
        ActorSystem::cancel_token(self.as_ref())
    }
}
