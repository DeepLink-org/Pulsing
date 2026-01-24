//! Actor System Extension Traits
//!
//! This module defines the public API surface for ActorSystem through traits:
//! - [`ActorSystemCoreExt`] - Core spawn and resolve operations (primary API)
//! - [`ActorSystemAdvancedExt`] - Factory-based spawning for supervision/restart
//! - [`ActorSystemOpsExt`] - Operations, introspection, and lifecycle management

use std::net::SocketAddr;
use std::sync::Arc;

use crate::actor::{Actor, ActorId, ActorPath, ActorRef, IntoActorPath, NodeId};
use crate::cluster::{MemberInfo, NamedActorInfo};
use crate::system_actor::BoxedActorFactory;

use super::config::{ResolveOptions, SpawnOptions};
use super::NodeLoadTracker;

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
/// - [`spawn`](Self::spawn) - Spawn an actor with a local name
/// - [`spawn_with_options`](Self::spawn_with_options) - Spawn with custom options
/// - [`spawn_named`](Self::spawn_named) - Spawn a publicly discoverable named actor
/// - [`spawn_named_with_options`](Self::spawn_named_with_options) - Spawn named with custom options
///
/// # Resolve Methods
/// - [`actor_ref`](Self::actor_ref) - Get ActorRef by ActorId
/// - [`resolve_named`](Self::resolve_named) - Resolve a named actor by path
/// - [`resolve_named_with_options`](Self::resolve_named_with_options) - Resolve with load balancing/filtering
/// - [`resolve_named_lazy`](Self::resolve_named_lazy) - Lazy resolution with auto-refresh
///
/// # Example
/// ```rust,ignore
/// use pulsing_actor::prelude::*;
///
/// let system = ActorSystem::builder().build().await?;
///
/// // Spawn a local actor
/// let actor = system.spawn("my_actor", MyActor::new()).await?;
///
/// // Spawn a named actor (discoverable across cluster)
/// let named = system.spawn_named("services/echo", "echo", EchoActor).await?;
///
/// // Resolve by name
/// let resolved = system.resolve_named("services/echo", None).await?;
/// ```
#[async_trait::async_trait]
pub trait ActorSystemCoreExt {
    /// Spawn an actor with a local name (uses system default mailbox capacity)
    async fn spawn<A>(&self, name: impl AsRef<str> + Send, actor: A) -> anyhow::Result<ActorRef>
    where
        A: Actor;

    /// Spawn an actor with custom options
    async fn spawn_with_options<A>(
        &self,
        name: impl AsRef<str> + Send,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor;

    /// Spawn a named actor (publicly accessible via named path)
    ///
    /// Named actors are discoverable across the cluster by their path.
    ///
    /// # Arguments
    /// - `path` - The public path for discovery (e.g., "services/echo")
    /// - `local_name` - The local name for this instance
    /// - `actor` - The actor instance
    async fn spawn_named<P, A>(
        &self,
        path: P,
        local_name: impl AsRef<str> + Send,
        actor: A,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
        A: Actor;

    /// Spawn a named actor with custom options
    async fn spawn_named_with_options<P, A>(
        &self,
        path: P,
        local_name: impl AsRef<str> + Send,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
        A: Actor;

    /// Get ActorRef for a local or remote actor by ID
    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef>;

    /// Resolve a named actor by path
    ///
    /// Returns an ActorRef that points to the current location of the named actor.
    /// Note: If the actor migrates, this reference may become stale.
    /// For actors that may migrate, consider using [`resolve_named_lazy`](Self::resolve_named_lazy).
    async fn resolve_named<P>(&self, path: P, node_id: Option<&NodeId>) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send;

    /// Resolve a named actor with custom options (load balancing, health filtering)
    async fn resolve_named_with_options(
        &self,
        path: &ActorPath,
        options: ResolveOptions,
    ) -> anyhow::Result<ActorRef>;

    /// Resolve a named actor with lazy resolution (re-resolves after cache expires)
    ///
    /// Returns an ActorRef that automatically re-resolves after ~5 seconds.
    /// This is useful for named actors that may migrate between nodes.
    fn resolve_named_lazy<P>(&self, path: P) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath;
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
/// cannot be restarted. Use `spawn_factory` or `spawn_named_factory` if you need
/// supervision with restart capability.
///
/// # Example
/// ```rust,ignore
/// use pulsing_actor::prelude::*;
///
/// let system = ActorSystem::builder().build().await?;
///
/// // Spawn with factory - enables restart on failure
/// let options = SpawnOptions::new()
///     .supervision(SupervisionSpec::new()
///         .restart_policy(RestartPolicy::OnFailure)
///         .max_restarts(3));
///
/// let actor = system.spawn_factory("worker", || Ok(Worker::new()), options).await?;
/// ```
#[async_trait::async_trait]
pub trait ActorSystemAdvancedExt {
    /// Spawn an actor using a factory function (enables supervision restarts)
    async fn spawn_factory<F, A>(
        &self,
        name: impl AsRef<str> + Send,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor;

    /// Spawn a named actor using a factory function
    async fn spawn_named_factory<P, F, A>(
        &self,
        path: P,
        local_name: impl AsRef<str> + Send,
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
        A: Actor;

    /// Spawn an anonymous actor with custom options
    async fn spawn_anonymous_with_options<A>(
        &self,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor;

    /// Get load tracker for a node address
    fn get_node_load_tracker(&self, addr: &SocketAddr) -> Option<Arc<NodeLoadTracker>>;

    /// Decrement load after a request completes
    fn decrement_node_load(&self, addr: &SocketAddr);

    /// Resolve an actor address and get an ActorRef
    async fn resolve(&self, address: &crate::actor::ActorAddress) -> anyhow::Result<ActorRef>;

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
    async fn spawn<A>(&self, name: impl AsRef<str> + Send, actor: A) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        ActorSystem::spawn(self, name, actor).await
    }

    async fn spawn_with_options<A>(
        &self,
        name: impl AsRef<str> + Send,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        ActorSystem::spawn_with_options(self, name, actor, options).await
    }

    async fn spawn_named<P, A>(
        &self,
        path: P,
        local_name: impl AsRef<str> + Send,
        actor: A,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
        A: Actor,
    {
        ActorSystem::spawn_named(self, path, local_name, actor).await
    }

    async fn spawn_named_with_options<P, A>(
        &self,
        path: P,
        local_name: impl AsRef<str> + Send,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
        A: Actor,
    {
        ActorSystem::spawn_named_with_options(self, path, local_name, actor, options).await
    }

    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef> {
        ActorSystem::actor_ref(self.as_ref(), id).await
    }

    async fn resolve_named<P>(&self, path: P, node_id: Option<&NodeId>) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
    {
        ActorSystem::resolve_named(self.as_ref(), path, node_id).await
    }

    async fn resolve_named_with_options(
        &self,
        path: &ActorPath,
        options: ResolveOptions,
    ) -> anyhow::Result<ActorRef> {
        ActorSystem::resolve_named_with_options(self.as_ref(), path, options).await
    }

    fn resolve_named_lazy<P>(&self, path: P) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath,
    {
        ActorSystem::resolve_named_lazy(self, path)
    }
}

#[async_trait::async_trait]
impl ActorSystemAdvancedExt for Arc<ActorSystem> {
    async fn spawn_factory<F, A>(
        &self,
        name: impl AsRef<str> + Send,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor,
    {
        ActorSystem::spawn_factory(self, name, factory, options).await
    }

    async fn spawn_named_factory<P, F, A>(
        &self,
        path: P,
        local_name: impl AsRef<str> + Send,
        factory: F,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        P: IntoActorPath + Send,
        F: FnMut() -> anyhow::Result<A> + Send + 'static,
        A: Actor,
    {
        ActorSystem::spawn_named_factory(self, path, local_name, factory, options).await
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
        A: Actor,
    {
        ActorSystem::spawn_anonymous(self, actor).await
    }

    async fn spawn_anonymous_with_options<A>(
        &self,
        actor: A,
        options: SpawnOptions,
    ) -> anyhow::Result<ActorRef>
    where
        A: Actor,
    {
        ActorSystem::spawn_anonymous_with_options(self, actor, options).await
    }

    fn get_node_load_tracker(&self, addr: &SocketAddr) -> Option<Arc<NodeLoadTracker>> {
        ActorSystem::get_node_load_tracker(self.as_ref(), addr)
    }

    fn decrement_node_load(&self, addr: &SocketAddr) {
        ActorSystem::decrement_node_load(self.as_ref(), addr)
    }

    async fn resolve(&self, address: &crate::actor::ActorAddress) -> anyhow::Result<ActorRef> {
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
