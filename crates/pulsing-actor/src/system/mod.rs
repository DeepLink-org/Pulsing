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
mod lifecycle;
mod load_balancer;
mod resolve;
mod runtime;
mod spawn;
mod traits;

pub use config::{
    ActorSystemBuilder, ConfigValidationError, ResolveOptions, SpawnOptions, SystemConfig,
};
pub use handle::ActorStats;
pub use load_balancer::NodeLoadTracker;
pub use traits::{ActorSystemAdvancedExt, ActorSystemCoreExt, ActorSystemOpsExt};

use crate::actor::{ActorId, ActorPath, ActorRef, ActorResolver, ActorSystemRef, Envelope, NodeId};
use crate::cluster::{GossipBackend, HeadNodeBackend, NamingBackend};
use crate::policies::{LoadBalancingPolicy, RoundRobinPolicy};
use crate::system_actor::{BoxedActorFactory, SystemActor, SystemRef, SYSTEM_ACTOR_PATH};
use crate::transport::Http2Transport;
use crate::watch::ActorLifecycle;
use dashmap::DashMap;
use handle::LocalActorHandle;
use handler::SystemMessageHandler;
use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// The Actor System - manages actors and cluster membership
pub struct ActorSystem {
    /// Local node ID
    pub(crate) node_id: NodeId,

    /// HTTP/2 address
    pub(crate) addr: SocketAddr,

    /// Default mailbox capacity for actors
    pub(crate) default_mailbox_capacity: usize,

    /// Local actors indexed by local_id (O(1) lookup by ActorId)
    pub(crate) local_actors: Arc<DashMap<u64, LocalActorHandle>>,

    /// Actor name to local_id mapping (for name-based lookups)
    pub(crate) actor_names: Arc<DashMap<String, u64>>,

    /// Named actor path to local actor name mapping (path_string -> actor_name)
    pub(crate) named_actor_paths: Arc<DashMap<String, String>>,

    /// Naming backend (for discovery)
    pub(crate) cluster: Arc<RwLock<Option<Arc<dyn NamingBackend>>>>,

    /// HTTP/2 transport
    pub(crate) transport: Arc<Http2Transport>,

    /// Cancellation token
    pub(crate) cancel_token: CancellationToken,

    /// Actor lifecycle manager (watch, termination handling)
    pub(crate) lifecycle: Arc<ActorLifecycle>,

    /// Actor ID counter (for generating unique local IDs)
    pub(crate) actor_id_counter: AtomicU64,

    /// Default load balancing policy
    pub(crate) default_lb_policy: Arc<dyn LoadBalancingPolicy>,

    /// Per-node load tracking for remote nodes
    pub(crate) node_load: Arc<DashMap<SocketAddr, Arc<NodeLoadTracker>>>,
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
            local_actors,
            actor_names,
            named_actor_paths,
            cluster: cluster_holder,
            transport,
            cancel_token,
            lifecycle,
            actor_id_counter: AtomicU64::new(1), // Start from 1 (0 reserved for system)
            default_lb_policy: Arc::new(RoundRobinPolicy::new()),
            node_load: Arc::new(DashMap::new()),
        });

        // Start SystemActor
        system.start_system_actor().await?;

        Ok(system)
    }

    /// Start SystemActor (internal, called during system creation)
    async fn start_system_actor(self: &Arc<Self>) -> anyhow::Result<()> {
        // Create senders snapshot for SystemRef
        let local_actor_senders: Arc<DashMap<String, mpsc::Sender<Envelope>>> =
            Arc::new(DashMap::new());
        for entry in self.local_actors.iter() {
            // Find name for this actor (reverse lookup from actor_names)
            if let Some(name_entry) = self.actor_names.iter().find(|e| *e.value() == *entry.key()) {
                local_actor_senders.insert(name_entry.key().clone(), entry.sender.clone());
            }
        }

        // Create named_actor_paths snapshot
        let named_actor_paths: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
        for entry in self.named_actor_paths.iter() {
            named_actor_paths.insert(entry.key().clone(), entry.value().clone());
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
