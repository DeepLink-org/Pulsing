//! Naming backend trait.

use crate::actor::{ActorId, ActorPath, NodeId, StopReason};
use crate::cluster::member::{MemberInfo, NamedActorInfo, NamedActorInstance};
use crate::error::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::net::SocketAddr;
use tokio_util::sync::CancellationToken;

/// Trait for naming backends that provide cluster membership and actor discovery.
#[async_trait]
pub trait NamingBackend: Send + Sync {
    async fn join(&self, seeds: Vec<SocketAddr>) -> Result<()>;

    async fn leave(&self) -> Result<()>;

    async fn all_members(&self) -> Vec<MemberInfo>;

    async fn alive_members(&self) -> Vec<MemberInfo>;

    async fn get_member(&self, node_id: &NodeId) -> Option<MemberInfo>;

    async fn register_named_actor(&self, path: ActorPath);

    async fn register_named_actor_full(
        &self,
        path: ActorPath,
        actor_id: ActorId,
        metadata: HashMap<String, String>,
    );

    async fn unregister_named_actor(&self, path: &ActorPath);

    async fn broadcast_named_actor_failed(&self, path: &ActorPath, reason: &StopReason);

    async fn lookup_named_actor(&self, path: &ActorPath) -> Option<NamedActorInfo>;

    async fn select_named_actor_instance(&self, path: &ActorPath) -> Option<MemberInfo>;

    async fn get_named_actor_instances(&self, path: &ActorPath) -> Vec<MemberInfo>;

    async fn get_named_actor_instances_detailed(
        &self,
        path: &ActorPath,
    ) -> Vec<(MemberInfo, Option<NamedActorInstance>)>;

    async fn all_named_actors(&self) -> Vec<NamedActorInfo>;

    async fn register_actor(&self, actor_id: ActorId);

    async fn unregister_actor(&self, actor_id: &ActorId);

    async fn lookup_actor(&self, actor_id: &ActorId) -> Option<MemberInfo>;

    fn start(&self, cancel: CancellationToken);

    fn local_node(&self) -> &NodeId;

    fn local_addr(&self) -> SocketAddr;

    fn as_any(&self) -> &dyn std::any::Any;
}
