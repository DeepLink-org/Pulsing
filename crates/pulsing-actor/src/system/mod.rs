//! Actor System - the main entry point for creating and managing actors
//!
//! This module provides:
//! - [`ActorSystem`] - The main system for managing actors
//! - [`SystemConfig`] - Configuration for the actor system
//! - [`SpawnOptions`] - Options for spawning actors
//! - [`ResolveOptions`] - Options for resolving named actors
//!
//! # Examples
//!
//! ## Creating a Standalone System
//!
//! For single-node development and testing:
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! // Create a standalone system (no network)
//! let system = ActorSystem::new(SystemConfig::standalone()).await?;
//!
//! // The system is ready to spawn actors
//! println!("System started on node: {}", system.node_id());
//!
//! // Clean shutdown when done
//! system.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Creating a Cluster Node
//!
//! For production multi-node deployment:
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! // Seed node (first node in cluster)
//! let addr: std::net::SocketAddr = "0.0.0.0:8000".parse()?;
//! let config = SystemConfig::with_addr(addr);
//! let seed_system = ActorSystem::new(config).await?;
//!
//! // Worker node joining the cluster
//! let addr: std::net::SocketAddr = "0.0.0.0:8001".parse()?;
//! let seed: std::net::SocketAddr = "127.0.0.1:8000".parse()?;
//! let config = SystemConfig::with_addr(addr)
//!     .with_seeds(vec![seed]);
//! let worker_system = ActorSystem::new(config).await?;
//!
//! println!("Cluster formed with 2 nodes");
//! # Ok(())
//! # }
//! ```
//!
//! ## Listing Local Actors
//!
//! ```no_run
//! use pulsing_actor::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! # let system = ActorSystem::new(SystemConfig::standalone()).await?;
//! // Get all named actors in this system
//! let names = system.local_actor_names();
//! for name in names {
//!     println!("Actor: {}", name);
//! }
//! # Ok(())
//! # }
//! ```

mod config;
mod handle;
mod handler;
mod lifecycle;
mod load_balancer;
pub mod registry;
mod resolve;
mod runtime;
mod spawn;
mod traits;

pub use config::{
    ActorSystemBuilder, ConfigValidationError, ResolveOptions, SpawnOptions, SystemConfig,
};
pub use handle::ActorStats;
pub use load_balancer::NodeLoadTracker;
pub use registry::ActorRegistry;
pub use traits::{ActorSystemCoreExt, ActorSystemOpsExt};

use crate::actor::{
    ActorId, ActorPath, ActorRef, ActorResolver, ActorSystemRef, Message, NodeId, StopReason,
};
use crate::cluster::{GossipBackend, HeadNodeBackend, NamingBackend};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::performance_store::{
    PerformanceSnapshot, PerformanceStore, DEFAULT_PERFORMANCE_HISTORY_CAPACITY,
};
use crate::policies::{LoadBalancingPolicy, RoundRobinPolicy};
use crate::system_actor::{
    BoxedActorFactory, DefaultActorFactory, NodeLifecycle, NodeState, ShmManager, SystemActor,
    SystemHost, SystemMessage, SystemRef, SystemResponse, SYSTEM_ACTOR_PATH,
};
use crate::transport::Http2Transport;
use dashmap::DashMap;
use handler::SystemMessageHandler;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// The Actor System - manages actors and cluster membership.
///
/// Actor management (spawn, name lookup, lifecycle) is delegated to
/// [`ActorRegistry`]. Transport, cluster, and load balancing remain here.
pub struct ActorSystem {
    /// Local node ID
    pub(crate) node_id: NodeId,

    /// HTTP/2 address
    pub(crate) addr: SocketAddr,

    /// Default mailbox capacity for actors
    pub(crate) default_mailbox_capacity: usize,

    /// Actor registry: manages local actors, names, paths, lifecycle
    pub(crate) registry: Arc<ActorRegistry>,

    /// Naming backend (for discovery)
    pub(crate) cluster: Arc<RwLock<Option<Arc<dyn NamingBackend>>>>,

    /// HTTP/2 transport
    pub(crate) transport: Arc<Http2Transport>,

    /// Cancellation token
    pub(crate) cancel_token: CancellationToken,

    /// Default load balancing policy
    pub(crate) default_lb_policy: Arc<dyn LoadBalancingPolicy>,

    /// Per-node load tracking for remote nodes
    pub(crate) node_load: Arc<DashMap<SocketAddr, Arc<NodeLoadTracker>>>,

    /// Shared [`SystemActor`] monitoring registry + metrics (filled when `system/core` starts).
    pub(crate) system_monitor: Arc<
        OnceLock<(
            Arc<crate::system_actor::SystemMetrics>,
            Arc<crate::system_actor::ActorRegistry>,
        )>,
    >,

    /// Ring buffer of recent `GetMetrics` snapshots (for local analysis / SQL adapters).
    pub(crate) performance_store: Arc<PerformanceStore>,

    /// Node-level shared-memory control plane used by future same-host tensor backends.
    pub(crate) shm_manager: Arc<ShmManager>,

    /// Authoritative node control-plane lifecycle.
    pub(crate) node_lifecycle: Arc<NodeLifecycle>,
}

impl ActorSystem {
    const BOOTSTRAP_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

    /// Create a builder for configuring ActorSystem
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let system = ActorSystem::builder().build().await?;
    /// ```
    pub fn builder() -> ActorSystemBuilder {
        ActorSystemBuilder::default()
    }

    /// Create a new actor system
    pub async fn new(config: SystemConfig) -> Result<Arc<Self>> {
        Self::new_inner(config, None).await
    }

    /// Create an actor system with a custom actors-system service factory.
    ///
    /// The factory is installed while `system/core` is bootstrapped, before
    /// the node becomes usable.  This is the supported replacement for trying
    /// to replace an already-running SystemActor after [`Self::new`].
    pub async fn new_with_system_actor_factory(
        config: SystemConfig,
        factory: BoxedActorFactory,
    ) -> Result<Arc<Self>> {
        Self::new_inner(config, Some(factory)).await
    }

    async fn new_inner(
        config: SystemConfig,
        system_actor_factory: Option<BoxedActorFactory>,
    ) -> Result<Arc<Self>> {
        let cancel_token = CancellationToken::new();
        let node_id = NodeId::generate();
        let registry = Arc::new(ActorRegistry::new());
        let cluster_holder: Arc<RwLock<Option<Arc<dyn NamingBackend>>>> =
            Arc::new(RwLock::new(None));
        let system_monitor: Arc<
            OnceLock<(
                Arc<crate::system_actor::SystemMetrics>,
                Arc<crate::system_actor::ActorRegistry>,
            )>,
        > = Arc::new(OnceLock::new());
        let performance_store =
            Arc::new(PerformanceStore::new(DEFAULT_PERFORMANCE_HISTORY_CAPACITY));
        let shm_manager = Arc::new(ShmManager::new());
        let node_lifecycle = Arc::new(NodeLifecycle::new());

        // Create message handler (needs registry and cluster reference)
        let handler = SystemMessageHandler::new(
            node_id,
            registry.clone(),
            cluster_holder.clone(),
            system_monitor.clone(),
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

        // Join cluster if seed nodes provided (only for gossip mode).
        let join_result = if !config.seed_nodes.is_empty()
            && config.head_addr.is_none()
            && !config.is_head_node
        {
            backend.join(config.seed_nodes).await
        } else if config.head_addr.is_some() || config.is_head_node {
            // For head node mode, join is handled internally.
            backend.join(Vec::new()).await
        } else {
            Ok(())
        };
        if let Err(error) = join_result {
            Self::cleanup_failed_bootstrap(
                &node_lifecycle,
                &shm_manager,
                &cancel_token,
                backend.leave(),
                Self::BOOTSTRAP_CLEANUP_TIMEOUT,
            )
            .await;
            return Err(error);
        }

        crate::actor_store::init_actor_memtable();
        crate::metrics_store::init_metrics_memtable();
        crate::members_store::init_members_memtable();

        let system = Arc::new(Self {
            node_id,
            addr: actual_addr,
            default_mailbox_capacity: config.default_mailbox_capacity,
            registry,
            cluster: cluster_holder,
            transport,
            cancel_token,
            default_lb_policy: Arc::new(RoundRobinPolicy::new()),
            node_load: Arc::new(DashMap::new()),
            system_monitor,
            performance_store,
            shm_manager,
            node_lifecycle,
        });

        // SystemRoot is part of bootstrap, not a runtime replacement.  This
        // keeps factory/service dependencies fixed before the node is ready.
        let root_result = if let Some(factory) = system_actor_factory {
            system.start_system_actor_with_factory(factory).await
        } else {
            system.start_system_actor().await
        };
        if let Err(error) = root_result {
            Self::cleanup_failed_bootstrap(
                &system.node_lifecycle,
                &system.shm_manager,
                &system.cancel_token,
                backend.leave(),
                Self::BOOTSTRAP_CLEANUP_TIMEOUT,
            )
            .await;
            return Err(error);
        }

        crate::members_store::upsert_member(
            node_id,
            actual_addr,
            crate::cluster::NodeStatus::Online,
            0,
        );

        Ok(system)
    }

    async fn cleanup_failed_bootstrap<F>(
        lifecycle: &NodeLifecycle,
        shm_manager: &ShmManager,
        cancel_token: &CancellationToken,
        leave: F,
        leave_timeout: Duration,
    ) where
        F: Future<Output = Result<()>>,
    {
        let _ = lifecycle.transition(NodeState::Failed);
        match tokio::time::timeout(leave_timeout, leave).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(%error, "Failed to leave cluster during bootstrap cleanup");
            }
            Err(_) => {
                tracing::warn!(
                    timeout_ms = leave_timeout.as_millis() as u64,
                    "Timed out leaving cluster during bootstrap cleanup"
                );
            }
        }
        cancel_token.cancel();
        shm_manager.clear();
    }

    /// Start SystemActor (internal, called during system creation)
    async fn start_system_actor(self: &Arc<Self>) -> Result<()> {
        self.spawn_system_actor(Box::new(DefaultActorFactory)).await
    }

    /// Start SystemActor with a custom factory during bootstrap.
    ///
    /// `ActorSystem::new` already starts `system/core`; callers that need a
    /// custom factory must use [`Self::new_with_system_actor_factory`].  This
    /// method remains public for compatibility with custom bootstrap code.
    pub async fn start_system_actor_with_factory(
        self: &Arc<Self>,
        factory: BoxedActorFactory,
    ) -> Result<()> {
        // Check if already started
        if self.registry.has_name(SYSTEM_ACTOR_PATH) {
            return Err(PulsingError::from(RuntimeError::Other(
                "SystemActor already started".into(),
            )));
        }

        self.spawn_system_actor(factory).await
    }

    async fn spawn_system_actor(self: &Arc<Self>, factory: BoxedActorFactory) -> Result<()> {
        let system_ref = self.system_ref_compat();
        let metrics = Arc::new(crate::system_actor::SystemMetrics::new());
        let legacy_registry = Arc::new(crate::system_actor::ActorRegistry::new());
        let host = SystemHost::hosted(
            &system_ref,
            Arc::downgrade(self),
            metrics.clone(),
            self.performance_store.clone(),
            self.shm_manager.clone(),
            self.node_lifecycle.clone(),
        );
        let system_actor =
            SystemActor::new_hosted(factory, legacy_registry.clone(), metrics.clone(), host);

        let system_path = ActorPath::new_system(SYSTEM_ACTOR_PATH)?;
        let actor_ref = self
            .spawning()
            .path(system_path)
            .defer_cluster_publication()
            .spawn(system_actor)
            .await?;

        if let Err(error) = self.await_system_actor_ready(&actor_ref).await {
            let _ = self.node_lifecycle.transition(NodeState::Failed);
            let _ = self.stop(SYSTEM_ACTOR_PATH).await;
            return Err(error);
        }
        if let Some(cluster) = self.cluster.read().await.as_ref() {
            let path = ActorPath::new_system(SYSTEM_ACTOR_PATH)?;
            cluster.register_named_actor(path).await;
            cluster.register_actor(*actor_ref.id()).await;
        }
        let _ = self.system_monitor.set((metrics, legacy_registry));
        tracing::debug!(path = SYSTEM_ACTOR_PATH, "SystemActor ready");
        Ok(())
    }

    fn system_ref_compat(&self) -> Arc<SystemRef> {
        Arc::new(SystemRef::new(self.node_id, self.addr))
    }

    async fn await_system_actor_ready(&self, actor_ref: &ActorRef) -> Result<()> {
        let data = serde_json::to_vec(&SystemMessage::Ping)
            .map_err(|error| PulsingError::from(RuntimeError::Serialization(error.to_string())))?;
        let request = Message::Single {
            msg_type: "SystemMessage".to_string(),
            data,
        };
        let response =
            tokio::time::timeout(std::time::Duration::from_secs(5), actor_ref.send(request))
                .await
                .map_err(|_| {
                    PulsingError::from(RuntimeError::Other(
                        "SystemActor readiness timed out".to_string(),
                    ))
                })??;
        let response: SystemResponse = response.parse()?;
        if matches!(response, SystemResponse::Pong { .. })
            && self.node_lifecycle.state() == NodeState::Ready
        {
            Ok(())
        } else {
            Err(PulsingError::from(RuntimeError::Other(format!(
                "SystemActor did not become ready: {response:?}"
            ))))
        }
    }

    /// Get SystemActor reference
    pub async fn system(&self) -> Result<ActorRef> {
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
        self.registry.actor_names_list()
    }

    /// Get a local actor reference by name
    ///
    /// Returns None if the actor doesn't exist locally.
    /// This is an O(1) operation.
    pub fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef> {
        self.registry.local_actor_ref_by_name(name)
    }

    /// Recent metric snapshots (newest first), recorded on each `GetMetrics` to `system/core`.
    pub fn performance_recent(&self, limit: usize) -> Vec<PerformanceSnapshot> {
        self.performance_store.recent(limit)
    }

    /// In-memory performance history store (same instance as [`SystemActor`] uses).
    pub fn performance_store(&self) -> &Arc<PerformanceStore> {
        &self.performance_store
    }

    /// Node-scoped shared-memory control plane.
    ///
    /// The initial backend is in-process and provides the stable region/lease
    /// contract required before enabling cross-process mappings.
    pub fn shm_manager(&self) -> &Arc<ShmManager> {
        &self.shm_manager
    }

    /// Keep [`SystemActor`] `GetMetrics` / list in sync with real spawns (excludes `system/core`).
    pub(crate) fn notify_monitor_actor_spawned(
        &self,
        name_str: &Option<String>,
        actor_id: ActorId,
        actor_metadata: &std::collections::HashMap<String, String>,
    ) {
        if name_str.as_deref() == Some(SYSTEM_ACTOR_PATH) {
            return;
        }
        let key = match name_str {
            Some(n) => n.clone(),
            None => actor_id.to_string(),
        };

        crate::actor_store::write_actor_spawned(&key, actor_id, self.node_id, actor_metadata);

        let Some((metrics, reg)) = self.system_monitor.get() else {
            return;
        };
        let actor_type = actor_metadata
            .get("class")
            .or_else(|| actor_metadata.get("type"))
            .cloned()
            .unwrap_or_else(|| "Actor".to_string());
        reg.register_with_metadata(&key, actor_id, &actor_type, actor_metadata.clone());
        metrics.inc_actor_created();
    }

    pub(crate) fn notify_monitor_actor_stopped(
        &self,
        actor_name: &str,
        actor_id: ActorId,
        reason: &StopReason,
    ) {
        if actor_name == SYSTEM_ACTOR_PATH {
            return;
        }

        crate::actor_store::write_actor_stopped(actor_name, actor_id, self.node_id, reason);

        let Some((metrics, reg)) = self.system_monitor.get() else {
            return;
        };
        if reg.unregister(actor_name).is_some() {
            metrics.inc_actor_stopped();
        }
    }
}

#[async_trait::async_trait]
impl ActorSystemRef for ActorSystem {
    async fn actor_ref(&self, id: &ActorId) -> Result<ActorRef> {
        ActorSystem::actor_ref(self, id).await
    }

    fn node_id(&self) -> NodeId {
        self.node_id
    }

    async fn watch(&self, watcher: &ActorId, target: &ActorId) -> Result<()> {
        // Check if target is a local actor
        if self.registry.get_handle(target).is_none() {
            return Err(PulsingError::from(RuntimeError::Other(format!(
                "Cannot watch remote actor: {} (watching remote actors not yet supported)",
                target
            ))));
        }

        self.registry.lifecycle.watch(watcher, target).await;
        Ok(())
    }

    async fn unwatch(&self, watcher: &ActorId, target: &ActorId) -> Result<()> {
        self.registry.lifecycle.unwatch(watcher, target).await;
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
    async fn resolve_path(&self, path: &ActorPath) -> Result<ActorRef> {
        // Use direct resolution (not lazy) to avoid infinite recursion
        self.resolve_named_direct(path, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn failed_bootstrap_cleanup_leaves_then_cancels_and_clears_host_resources() {
        let lifecycle = NodeLifecycle::new();
        let shm_manager = ShmManager::new();
        let descriptor =
            shm_manager.offer(Bytes::from_static(b"bootstrap"), Duration::from_secs(30));
        let cancel_token = CancellationToken::new();
        let leave_called = Arc::new(AtomicBool::new(false));
        let leave_called_in_future = leave_called.clone();
        let cancel_observed_by_leave = cancel_token.clone();

        ActorSystem::cleanup_failed_bootstrap(
            &lifecycle,
            &shm_manager,
            &cancel_token,
            async move {
                assert!(!cancel_observed_by_leave.is_cancelled());
                leave_called_in_future.store(true, Ordering::Release);
                Ok(())
            },
            Duration::from_millis(100),
        )
        .await;

        assert!(leave_called.load(Ordering::Acquire));
        assert!(cancel_token.is_cancelled());
        assert_eq!(lifecycle.state(), NodeState::Failed);
        assert_eq!(
            shm_manager.stats(),
            crate::system_actor::ShmStats::default()
        );
        assert!(shm_manager.map(&descriptor).is_err());
    }

    #[tokio::test]
    async fn failed_bootstrap_cleanup_continues_when_leave_fails() {
        let lifecycle = NodeLifecycle::new();
        let shm_manager = ShmManager::new();
        shm_manager.offer(Bytes::from_static(b"bootstrap"), Duration::from_secs(30));
        let cancel_token = CancellationToken::new();

        ActorSystem::cleanup_failed_bootstrap(
            &lifecycle,
            &shm_manager,
            &cancel_token,
            async {
                Err(PulsingError::from(RuntimeError::Other(
                    "injected leave failure".to_string(),
                )))
            },
            Duration::from_millis(100),
        )
        .await;

        assert!(cancel_token.is_cancelled());
        assert_eq!(lifecycle.state(), NodeState::Failed);
        assert_eq!(
            shm_manager.stats(),
            crate::system_actor::ShmStats::default()
        );
    }
}
