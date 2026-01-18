//! Naming backend trait for abstracting different naming service implementations
//!
//! This trait provides a unified interface for:
//! - Node discovery and membership management
//! - Named actor registration and discovery
//! - Actor location queries

use crate::actor::{ActorId, ActorPath, NodeId, StopReason};
use crate::cluster::member::{MemberInfo, NamedActorInfo, NamedActorInstance};
use async_trait::async_trait;
use std::collections::HashMap;
use std::net::SocketAddr;
use tokio_util::sync::CancellationToken;

/// Trait for naming backends that provide cluster membership and actor discovery
#[async_trait]
pub trait NamingBackend: Send + Sync {
    // ========================================================================
    // Node Management
    // ========================================================================

    /// Join the cluster via seed nodes
    async fn join(&self, seeds: Vec<SocketAddr>) -> anyhow::Result<()>;

    /// Leave the cluster gracefully
    async fn leave(&self) -> anyhow::Result<()>;

    /// Get all cluster members
    async fn all_members(&self) -> Vec<MemberInfo>;

    /// Get only alive cluster members
    async fn alive_members(&self) -> Vec<MemberInfo>;

    /// Get member information for a specific node
    async fn get_member(&self, node_id: &NodeId) -> Option<MemberInfo>;

    // ========================================================================
    // Named Actor Registration
    // ========================================================================

    /// Register a named actor (legacy, without actor_id)
    async fn register_named_actor(&self, path: ActorPath);

    /// Register a named actor with full details (actor_id and metadata)
    async fn register_named_actor_full(
        &self,
        path: ActorPath,
        actor_id: ActorId,
        metadata: HashMap<String, String>,
    );

    /// Unregister a named actor
    async fn unregister_named_actor(&self, path: &ActorPath);

    /// Broadcast that a named actor has failed
    async fn broadcast_named_actor_failed(&self, path: &ActorPath, reason: &StopReason);

    // ========================================================================
    // Named Actor Queries
    // ========================================================================

    /// Lookup named actor information
    async fn lookup_named_actor(&self, path: &ActorPath) -> Option<NamedActorInfo>;

    /// Select a named actor instance (for load balancing)
    async fn select_named_actor_instance(&self, path: &ActorPath) -> Option<MemberInfo>;

    /// Get all instances of a named actor
    async fn get_named_actor_instances(&self, path: &ActorPath) -> Vec<MemberInfo>;

    /// Get detailed instance information for a named actor
    async fn get_named_actor_instances_detailed(
        &self,
        path: &ActorPath,
    ) -> Vec<(MemberInfo, Option<NamedActorInstance>)>;

    /// Get all named actors in the cluster
    async fn all_named_actors(&self) -> Vec<NamedActorInfo>;

    // ========================================================================
    // Actor Registration (optional, some backends may not support)
    // ========================================================================

    /// Register an actor (for non-named actors)
    async fn register_actor(&self, actor_id: ActorId);

    /// Unregister an actor
    async fn unregister_actor(&self, actor_id: &ActorId);

    /// Lookup actor location
    async fn lookup_actor(&self, actor_id: &ActorId) -> Option<MemberInfo>;

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    /// Start the backend (e.g., start background tasks)
    fn start(&self, cancel: CancellationToken);

    /// Get the local node ID
    fn local_node(&self) -> &NodeId;

    /// Get the local node address
    fn local_addr(&self) -> SocketAddr;

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}
