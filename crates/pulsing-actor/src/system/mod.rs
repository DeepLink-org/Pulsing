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

use crate::actor::{ActorId, ActorPath, ActorRef, ActorResolver, ActorSystemRef, Envelope, NodeId};
use crate::cluster::{GossipBackend, HeadNodeBackend, NamingBackend};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::policies::{LoadBalancingPolicy, RoundRobinPolicy};
use crate::system_actor::{BoxedActorFactory, SystemActor, SystemRef, SYSTEM_ACTOR_PATH};
use crate::transport::Http2Transport;
use dashmap::DashMap;
use handler::SystemMessageHandler;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
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
        ActorSystemBuilder::default()
    }

    /// Create a new actor system
    pub async fn new(config: SystemConfig) -> Result<Arc<Self>> {
        let cancel_token = CancellationToken::new();
        let node_id = NodeId::generate();
        let registry = Arc::new(ActorRegistry::new());
        let cluster_holder: Arc<RwLock<Option<Arc<dyn NamingBackend>>>> =
            Arc::new(RwLock::new(None));

        // Create message handler (needs registry and cluster reference)
        let handler = SystemMessageHandler::new(node_id, registry.clone(), cluster_holder.clone());

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
            registry,
            cluster: cluster_holder,
            transport,
            cancel_token,
            default_lb_policy: Arc::new(RoundRobinPolicy::new()),
            node_load: Arc::new(DashMap::new()),
        });

        // Start SystemActor
        system.start_system_actor().await?;

        Ok(system)
    }

    /// Start SystemActor (internal, called during system creation)
    async fn start_system_actor(self: &Arc<Self>) -> Result<()> {
        // Create senders snapshot for SystemRef
        let local_actor_senders: Arc<DashMap<String, mpsc::Sender<Envelope>>> =
            Arc::new(DashMap::new());
        for entry in self.registry.iter_actors() {
            // Find name for this actor (reverse lookup from actor_names)
            if let Some(name_entry) = self
                .registry
                .actor_names
                .iter()
                .find(|e| *e.value() == *entry.key())
            {
                local_actor_senders.insert(name_entry.key().clone(), entry.sender.clone());
            }
        }

        // Create named_actor_paths snapshot
        let named_actor_paths: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
        for entry in self.registry.iter_named_paths() {
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
        self.spawning()
            .path(system_path)
            .spawn(system_actor)
            .await?;

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
    ) -> Result<()> {
        // Check if already started
        if self.registry.has_name(SYSTEM_ACTOR_PATH) {
            return Err(PulsingError::from(RuntimeError::Other(
                "SystemActor already started".into(),
            )));
        }

        // Create SystemRef (snapshot of named paths)
        let named_paths_snapshot: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
        for entry in self.registry.iter_named_paths() {
            named_paths_snapshot.insert(entry.key().clone(), entry.value().clone());
        }
        let system_ref = Arc::new(SystemRef {
            node_id: self.node_id,
            addr: self.addr,
            local_actors: Arc::new(DashMap::new()), // Will be updated
            named_actor_paths: named_paths_snapshot,
        });

        // Create SystemActor with custom factory
        let system_actor = SystemActor::new(system_ref, factory);

        // Spawn as named actor (use new_system to bypass namespace check)
        let system_path = ActorPath::new_system(SYSTEM_ACTOR_PATH)?;
        self.spawning()
            .path(system_path)
            .spawn(system_actor)
            .await?;

        tracing::debug!(
            path = SYSTEM_ACTOR_PATH,
            "SystemActor started with custom factory"
        );
        Ok(())
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
