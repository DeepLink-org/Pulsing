//! Actor resolution logic
//!
//! This module contains the implementation of actor resolution methods
//! that are used by the ActorSystem for locating actors by ID or name.

use crate::actor::{
    ActorAddress, ActorId, ActorPath, ActorRef, ActorResolver, IntoActorPath, NodeId,
};
use crate::cluster::{MemberInfo, MemberStatus, NamedActorInfo};
use crate::policies::LoadBalancingPolicy;
use crate::system::config::ResolveOptions;
use crate::system::load_balancer::{MemberWorker, NodeLoadTracker};
use crate::system::ActorSystem;
use crate::transport::Http2RemoteTransport;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

impl ActorSystem {
    async fn cluster_opt(&self) -> Option<Arc<dyn crate::cluster::NamingBackend>> {
        self.cluster.read().await.as_ref().cloned()
    }

    async fn cluster_or_err(&self) -> anyhow::Result<Arc<dyn crate::cluster::NamingBackend>> {
        self.cluster_opt()
            .await
            .ok_or_else(|| anyhow::anyhow!("Cluster not initialized"))
    }

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
        let cluster = self.cluster_or_err().await?;

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
    pub(crate) async fn resolve_named_direct(
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
        let cluster = self.cluster_or_err().await?;

        let instances = cluster.get_named_actor_instances(path).await;

        if instances.is_empty() {
            return Err(anyhow::anyhow!("Named actor not found: {}", path.as_str()));
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
            return Err(anyhow::anyhow!(
                "No healthy instances for named actor: {}",
                path.as_str()
            ));
        }

        let target = if let Some(nid) = options.node_id {
            healthy_instances
                .iter()
                .find(|i| i.node_id == nid)
                .ok_or_else(|| anyhow::anyhow!("Actor instance not found on node: {}", nid))?
        } else {
            let policy = options.policy.as_ref().unwrap_or(&self.default_lb_policy);
            self.select_instance(&healthy_instances, policy.as_ref())
        };

        if target.node_id == self.node_id {
            let actor_name = self
                .named_actor_paths
                .get(&path.as_str())
                .ok_or_else(|| anyhow::anyhow!("Named actor not found locally"))?
                .clone();

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

        let transport =
            Http2RemoteTransport::new_named(self.transport.client(), target.addr, path.clone());

        let actor_id = ActorId::new(target.node_id, 0);
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
    ) -> anyhow::Result<Vec<ActorRef>> {
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
