//! Actor resolution logic
//!
//! This module contains the implementation of actor resolution methods
//! that are used by the ActorSystem for locating actors by ID or name.

use crate::actor::{
    ActorAddress, ActorId, ActorPath, ActorRef, ActorResolver, IntoActorPath, NodeId,
};
use crate::cluster::{MemberInfo, MemberStatus, NamedActorInfo};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::policies::LoadBalancingPolicy;
use crate::system::config::ResolveOptions;
use crate::system::load_balancer::{MemberWorker, NodeLoadTracker};
use crate::system::ActorSystem;
use crate::transport::{Http2RemoteTransport, TransportTarget};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

impl ActorSystem {
    async fn cluster_opt(&self) -> Option<Arc<dyn crate::cluster::NamingBackend>> {
        self.cluster.read().await.as_ref().cloned()
    }

    async fn cluster_or_err(&self) -> Result<Arc<dyn crate::cluster::NamingBackend>> {
        self.cluster_opt()
            .await
            .ok_or_else(|| PulsingError::from(RuntimeError::ClusterNotInitialized))
    }

    /// Get ActorRef for a local or remote actor by ID
    ///
    /// This is an O(1) operation for local actors using ActorId indexing.
    pub async fn actor_ref(&self, id: &ActorId) -> Result<ActorRef> {
        // Try local lookup first (O(1))
        if let Some(handle) = self.registry.get_handle(id) {
            return Ok(ActorRef::local(handle.actor_id, handle.sender.clone()));
        }

        // Not found locally - try remote lookup via cluster
        let cluster = self.cluster_or_err().await?;

        // Lookup actor location in cluster
        if let Some(member_info) = cluster.lookup_actor(id).await {
            let transport = Http2RemoteTransport::builder(
                self.transport.client(),
                member_info.addr,
                TransportTarget::ById(*id),
            )
            .build();
            return Ok(ActorRef::remote(*id, member_info.addr, Arc::new(transport)));
        }

        Err(PulsingError::from(RuntimeError::actor_not_found(
            id.to_string(),
        )))
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
    pub async fn resolve_named<P>(&self, path: P, node_id: Option<&NodeId>) -> Result<ActorRef>
    where
        P: IntoActorPath,
    {
        let path = path.into_actor_path()?;
        let options = if let Some(nid) = node_id {
            ResolveOptions::default().node_id(*nid)
        } else {
            ResolveOptions::default()
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
    pub fn resolve_named_lazy<P>(self: &Arc<Self>, path: P) -> Result<ActorRef>
    where
        P: IntoActorPath,
    {
        let path = path.into_actor_path()?;
        Ok(ActorRef::lazy(path, self.clone() as Arc<dyn ActorResolver>))
    }

    /// Internal: Direct resolution for ActorResolver trait
    pub(crate) async fn resolve_named_direct(
        &self,
        path: &ActorPath,
        node_id: Option<&NodeId>,
    ) -> Result<ActorRef> {
        let options = if let Some(nid) = node_id {
            ResolveOptions::default().node_id(*nid)
        } else {
            ResolveOptions::default()
        };
        self.resolve_named_with_options(path, options).await
    }

    /// Resolve a named actor with custom options (load balancing, health filtering)
    pub async fn resolve_named_with_options(
        &self,
        path: &ActorPath,
        options: ResolveOptions,
    ) -> Result<ActorRef> {
        let cluster = self.cluster_or_err().await?;

        let instances = cluster.get_named_actor_instances(path).await;

        // When node_id is specified but gossip hasn't propagated the path yet (e.g. Topic
        // subscriber on another node), resolve by target node directly instead of failing.
        if instances.is_empty() {
            if let Some(nid) = options.node_id {
                if nid != self.node_id {
                    if let Some(member) = cluster.get_member(&nid).await {
                        if !options.filter_alive || member.status == MemberStatus::Alive {
                            let transport = Http2RemoteTransport::builder(
                                self.transport.client(),
                                member.addr,
                                TransportTarget::Named(path.clone()),
                            )
                            .build();
                            let actor_id = ActorId::generate();
                            return Ok(ActorRef::remote(
                                actor_id,
                                member.addr,
                                Arc::new(transport),
                            ));
                        }
                    }
                }
            }
            return Err(PulsingError::from(RuntimeError::named_actor_not_found(
                path.as_str(),
            )));
        }

        let healthy_instances: Vec<_> = if options.filter_alive {
            instances
                .into_iter()
                .filter(|i| i.status == MemberStatus::Alive)
                .collect()
        } else {
            instances
        };

        if healthy_instances.is_empty() {
            return Err(PulsingError::from(RuntimeError::no_healthy_instances(
                path.as_str(),
            )));
        }

        let target = if let Some(nid) = options.node_id {
            healthy_instances
                .iter()
                .find(|i| i.node_id == nid)
                .ok_or_else(|| PulsingError::from(RuntimeError::node_not_found(nid.to_string())))?
        } else {
            let policy = options.policy.as_ref().unwrap_or(&self.default_lb_policy);
            self.select_instance(&healthy_instances, policy.as_ref())
        };

        if target.node_id == self.node_id {
            let actor_name = self
                .registry
                .get_actor_name_by_path(&path.as_str())
                .ok_or_else(|| {
                    PulsingError::from(RuntimeError::named_actor_not_found(path.as_str()))
                })?;

            let local_id = self.registry.get_actor_id(&actor_name).ok_or_else(|| {
                PulsingError::from(RuntimeError::actor_not_found(actor_name.clone()))
            })?;

            let handle = self.registry.get_handle(&local_id).ok_or_else(|| {
                PulsingError::from(RuntimeError::actor_not_found(actor_name.clone()))
            })?;

            return Ok(ActorRef::local(handle.actor_id, handle.sender.clone()));
        }

        let transport = Http2RemoteTransport::builder(
            self.transport.client(),
            target.addr,
            TransportTarget::Named(path.clone()),
        )
        .build();

        // For named actors, we don't have a specific ActorId until we resolve
        // Use a placeholder ID (this will be replaced when the actor is actually accessed)
        let actor_id = ActorId::generate();
        Ok(ActorRef::remote(actor_id, target.addr, Arc::new(transport)))
    }

    /// Select an instance using load balancing policy
    pub(crate) fn select_instance<'a>(
        &self,
        instances: &'a [MemberInfo],
        policy: &dyn LoadBalancingPolicy,
    ) -> &'a MemberInfo {
        let workers: Vec<Arc<dyn crate::policies::Worker>> = instances
            .iter()
            .map(|m| {
                let tracker = self
                    .node_load
                    .entry(m.addr)
                    .or_insert_with(|| Arc::new(NodeLoadTracker::new()))
                    .clone();
                Arc::new(MemberWorker::new(m, tracker)) as Arc<dyn crate::policies::Worker>
            })
            .collect();

        let idx = policy.select_worker(&workers, None).unwrap_or(0);

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
    pub async fn resolve(&self, address: &ActorAddress) -> Result<ActorRef> {
        match address {
            ActorAddress::Named { path, instance } => {
                self.resolve_named(path, instance.as_ref()).await
            }
            ActorAddress::Global { actor_id, .. } => {
                // actor_id is already a full ActorId (u128)
                self.actor_ref(actor_id).await
            }
        }
    }

    /// Get all instances of a named actor across the cluster
    pub async fn get_named_instances(&self, path: &ActorPath) -> Vec<MemberInfo> {
        match self.cluster_opt().await {
            Some(cluster) => cluster.get_named_actor_instances(path).await,
            None => Vec::new(),
        }
    }

    /// Resolve all instances of a named actor as ActorRefs
    pub async fn resolve_all_instances(
        &self,
        path: &ActorPath,
        filter_alive: bool,
    ) -> Result<Vec<ActorRef>> {
        let cluster = self.cluster_or_err().await?;

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
        match self.cluster_opt().await {
            Some(cluster) => cluster.get_named_actor_instances_detailed(path).await,
            None => Vec::new(),
        }
    }

    /// Lookup named actor information
    pub async fn lookup_named(&self, path: &ActorPath) -> Option<NamedActorInfo> {
        match self.cluster_opt().await {
            Some(cluster) => cluster.lookup_named_actor(path).await,
            None => None,
        }
    }

    /// Get cluster member information
    pub async fn members(&self) -> Vec<MemberInfo> {
        match self.cluster_opt().await {
            Some(cluster) => cluster.all_members().await,
            None => Vec::new(),
        }
    }

    /// Get all named actors in the cluster
    pub async fn all_named_actors(&self) -> Vec<NamedActorInfo> {
        match self.cluster_opt().await {
            Some(cluster) => cluster.all_named_actors().await,
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::system::config::ActorSystemBuilder;

    #[tokio::test]
    async fn test_resolve_named_invalid_path_single_segment() {
        let system = ActorSystemBuilder::default()
            .addr("127.0.0.1:0")
            .build()
            .await
            .unwrap();
        let err = system.resolve_named("x", None).await.unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("namespace") || msg.contains("path"),
            "expected path/namespace error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_resolve_named_lazy_invalid_path() {
        let system = ActorSystemBuilder::default()
            .addr("127.0.0.1:0")
            .build()
            .await
            .unwrap();
        let err = system.resolve_named_lazy("single").unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("namespace") || msg.contains("path"),
            "expected path/namespace error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_resolve_named_valid_path_no_instances() {
        let system = ActorSystemBuilder::default()
            .addr("127.0.0.1:0")
            .build()
            .await
            .unwrap();
        let err = system
            .resolve_named("svc/nonexistent", None)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("nonexistent") || msg.contains("not_found") || msg.contains("not found"),
            "expected named_actor_not_found, got: {}",
            err
        );
    }
}
