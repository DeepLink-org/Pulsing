//! SystemActor - Built-in system actor, automatically started with each ActorSystem
//!
//! SystemActor provides the following features:
//! - Actor lifecycle management (create, stop, restart)
//! - System monitoring and metrics collection
//! - Cluster information query
//! - Extensible actor factory mechanism
//!
//! ## Named Path
//!
//! SystemActor has a fixed named path `system/core`, accessible via:
//!
//! ```ignore
//! // Local access
//! let sys = system.resolve_named(&ActorPath::new("system/core")?, None).await?;
//!
//! // Remote access
//! let remote_sys = system.resolve_named(&ActorPath::new("system/core")?, Some(&node_id)).await?;
//! ```

mod builtin;
mod factory;
mod host;
mod lifecycle;
mod messages;
mod service;
mod shm;

pub use factory::{ActorFactory, BoxedActorFactory, DefaultActorFactory};
pub(crate) use host::SystemHost;
pub(crate) use lifecycle::{NodeLifecycle, NodeState};
pub use messages::{ActorInfo, ActorStatusInfo, SystemMessage, SystemResponse};
pub use shm::{ShmBackend, ShmManager, ShmRegionDescriptor, ShmStats};

use crate::actor::{Actor, ActorContext, ActorId, Message};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::metrics::SystemMetrics as PrometheusSystemMetrics;
use crate::performance_store::PerformanceStore;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// Named path for SystemActor (system/core satisfies namespace/name format requirement)
pub const SYSTEM_ACTOR_PATH: &str = "system/core";

/// System metrics
#[derive(Debug, Default)]
pub struct SystemMetrics {
    /// Total messages processed
    messages_total: AtomicU64,
    /// Total actors created
    actors_created: AtomicU64,
    /// Total actors stopped
    actors_stopped: AtomicU64,
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inc_message(&self) {
        self.messages_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_actor_created(&self) {
        self.actors_created.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_actor_stopped(&self) {
        self.actors_stopped.fetch_add(1, Ordering::Relaxed);
    }

    pub fn messages_total(&self) -> u64 {
        self.messages_total.load(Ordering::Relaxed)
    }

    pub fn actors_created(&self) -> u64 {
        self.actors_created.load(Ordering::Relaxed)
    }

    pub fn actors_stopped(&self) -> u64 {
        self.actors_stopped.load(Ordering::Relaxed)
    }
}

/// Actor registry entry
struct ActorEntry {
    actor_id: ActorId,
    actor_type: String,
    created_at: Instant,
    /// Spawn-time metadata (e.g. Python ``class`` / ``module`` / ``file``).
    metadata: HashMap<String, String>,
}

/// Actor registry
pub struct ActorRegistry {
    /// name -> ActorEntry
    actors: DashMap<String, ActorEntry>,
}

impl ActorRegistry {
    pub fn new() -> Self {
        Self {
            actors: DashMap::new(),
        }
    }

    pub fn register(&self, name: &str, actor_id: ActorId, actor_type: &str) {
        self.register_with_metadata(name, actor_id, actor_type, HashMap::new());
    }

    pub fn register_with_metadata(
        &self,
        name: &str,
        actor_id: ActorId,
        actor_type: &str,
        metadata: HashMap<String, String>,
    ) {
        self.actors.insert(
            name.to_string(),
            ActorEntry {
                actor_id,
                actor_type: actor_type.to_string(),
                created_at: Instant::now(),
                metadata,
            },
        );
    }

    pub fn unregister(&self, name: &str) -> Option<ActorId> {
        self.actors.remove(name).map(|(_, e)| e.actor_id)
    }

    pub fn get(&self, name: &str) -> Option<ActorId> {
        self.actors.get(name).map(|e| e.actor_id)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.actors.contains_key(name)
    }

    pub fn count(&self) -> usize {
        self.actors.len()
    }

    pub fn list_all(&self) -> Vec<ActorInfo> {
        self.actors
            .iter()
            .map(|e| ActorInfo {
                name: e.key().clone(),
                actor_id: e.actor_id.0,
                actor_type: e.actor_type.clone(),
                uptime_secs: e.created_at.elapsed().as_secs(),
                metadata: e.metadata.clone(),
            })
            .collect()
    }

    pub fn get_info(&self, name: &str) -> Option<ActorInfo> {
        self.actors.get(name).map(|e| ActorInfo {
            name: name.to_string(),
            actor_id: e.actor_id.0,
            actor_type: e.actor_type.clone(),
            uptime_secs: e.created_at.elapsed().as_secs(),
            metadata: e.metadata.clone(),
        })
    }
}

impl Default for ActorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// System reference needed by SystemActor (avoids circular references)
pub struct SystemRef {
    /// Node ID
    pub node_id: crate::actor::NodeId,
    /// Local address
    pub addr: std::net::SocketAddr,
    /// Legacy spawn-time projection. System services use host capabilities and
    /// the authoritative ActorSystem registry instead.
    #[deprecated(
        since = "0.1.3",
        note = "use ActorSystem actor APIs; ActorSystem bootstrap no longer populates this snapshot"
    )]
    pub local_actors: Arc<DashMap<String, mpsc::Sender<crate::actor::Envelope>>>,
    /// Legacy spawn-time projection. System services use host capabilities and
    /// the authoritative ActorSystem registry instead.
    #[deprecated(
        since = "0.1.3",
        note = "use ActorSystem resolution APIs; ActorSystem bootstrap no longer populates this snapshot"
    )]
    pub named_actor_paths: Arc<DashMap<String, String>>,
}

impl SystemRef {
    /// Construct the compatibility reference required by manually hosted
    /// SystemActors. Actor and path snapshots are intentionally left empty.
    #[allow(deprecated)]
    pub fn new(node_id: crate::actor::NodeId, addr: std::net::SocketAddr) -> Self {
        Self {
            node_id,
            addr,
            local_actors: Arc::new(DashMap::new()),
            named_actor_paths: Arc::new(DashMap::new()),
        }
    }
}

/// SystemActor - Built-in system actor for each ActorSystem
pub struct SystemActor {
    /// Compatibility projection retained for public Rust/Python APIs.
    registry: Arc<ActorRegistry>,

    /// Compatibility counters shared with ActorSystem monitoring.
    metrics: Arc<SystemMetrics>,

    /// Bootstrap extension factory retained so compatibility builders can
    /// rebuild registrations before the actor starts.
    factory: Arc<dyn ActorFactory>,

    /// Narrow view of ActorSystem-owned resources. This is the only source used
    /// by built-in service handlers.
    host: host::SystemHost,

    /// Governed service registry owned by this SystemRoot.
    services: Arc<service::SystemServiceRegistry>,
}

impl SystemActor {
    /// Create a new SystemActor (private registry + metrics).
    pub fn new(system_ref: Arc<SystemRef>, factory: BoxedActorFactory) -> Self {
        Self::new_shared(
            system_ref,
            factory,
            Arc::new(ActorRegistry::new()),
            Arc::new(SystemMetrics::new()),
            Arc::new(PerformanceStore::new(256)),
        )
    }

    /// Create SystemActor with default factory
    pub fn with_default_factory(system_ref: Arc<SystemRef>) -> Self {
        Self::with_default_factory_shared(
            system_ref,
            Arc::new(ActorRegistry::new()),
            Arc::new(SystemMetrics::new()),
            Arc::new(PerformanceStore::new(256)),
        )
    }

    /// Shared registry + metrics (must match [`crate::system::ActorSystem::system_monitor`]).
    pub fn with_default_factory_shared(
        system_ref: Arc<SystemRef>,
        registry: Arc<ActorRegistry>,
        metrics: Arc<SystemMetrics>,
        performance_store: Arc<PerformanceStore>,
    ) -> Self {
        Self::new_shared(
            system_ref,
            Box::new(DefaultActorFactory),
            registry,
            metrics,
            performance_store,
        )
    }

    /// Shared registry + metrics with a custom factory.
    pub fn new_shared(
        system_ref: Arc<SystemRef>,
        factory: BoxedActorFactory,
        registry: Arc<ActorRegistry>,
        metrics: Arc<SystemMetrics>,
        performance_store: Arc<PerformanceStore>,
    ) -> Self {
        let host = host::SystemHost::standalone(
            &system_ref,
            registry.clone(),
            metrics.clone(),
            performance_store,
        );
        Self::from_host(factory, registry, metrics, host)
    }

    pub(crate) fn new_hosted(
        factory: BoxedActorFactory,
        registry: Arc<ActorRegistry>,
        metrics: Arc<SystemMetrics>,
        host: SystemHost,
    ) -> Self {
        Self::from_host(factory, registry, metrics, host)
    }

    fn from_host(
        factory: BoxedActorFactory,
        registry: Arc<ActorRegistry>,
        metrics: Arc<SystemMetrics>,
        host: host::SystemHost,
    ) -> Self {
        let factory: Arc<dyn ActorFactory> = Arc::from(factory);
        let services = Self::build_services(&host, factory.clone());
        Self {
            registry,
            metrics,
            factory,
            host,
            services,
        }
    }

    fn build_services(
        host: &host::SystemHost,
        factory: Arc<dyn ActorFactory>,
    ) -> Arc<service::SystemServiceRegistry> {
        let services = Arc::new(service::SystemServiceRegistry::new());
        for registration in builtin::registrations(host, factory) {
            services
                .register(registration)
                .expect("built-in system service manifests must be valid and unique");
        }
        services
    }

    fn rebuild_services(&mut self) {
        self.services = Self::build_services(&self.host, self.factory.clone());
    }

    /// Attach an authoritative host registry for read-only actor control.
    ///
    /// ActorSystem bootstrap uses [`Self::new_hosted`] so stop operations also
    /// receive the full lifecycle capability. This builder remains for source
    /// compatibility with manually constructed SystemActors.
    pub fn with_system_registry(
        mut self,
        reg: Arc<crate::system::registry::ActorRegistry>,
    ) -> Self {
        self.host = self
            .host
            .with_actor_control(Arc::new(host::RegistryActorControl::new(reg)));
        self.rebuild_services();
        self
    }

    /// Use the ActorSystem-owned shared-memory control plane.
    pub fn with_shm_manager(mut self, manager: Arc<ShmManager>) -> Self {
        self.host = self.host.with_shm_manager(manager);
        self.rebuild_services();
        self
    }

    /// Shared-memory control plane associated with this system actor.
    pub fn shm_manager(&self) -> &Arc<ShmManager> {
        self.host.shm_manager()
    }

    /// Get registry (for Python bindings)
    pub fn registry(&self) -> &Arc<ActorRegistry> {
        &self.registry
    }

    /// Get metrics
    pub fn metrics(&self) -> &Arc<SystemMetrics> {
        &self.metrics
    }

    /// Register a created actor (called externally)
    pub fn register_actor(&self, name: &str, actor_id: ActorId, actor_type: &str) {
        self.registry.register(name, actor_id, actor_type);
        self.metrics.inc_actor_created();
    }

    /// Unregister an actor (called externally)
    pub fn unregister_actor(&self, name: &str) {
        if self.registry.unregister(name).is_some() {
            self.metrics.inc_actor_stopped();
        }
    }

    /// Get Prometheus-compatible system metrics
    pub fn get_prometheus_metrics(&self) -> PrometheusSystemMetrics {
        builtin::prometheus_metrics(&self.host)
    }

    /// Generate JSON error response
    fn json_error_response(&self, message: &str) -> Result<Message> {
        let response = SystemResponse::Error {
            message: message.to_string(),
        };
        let json_data = serde_json::to_vec(&response)
            .map_err(|e| PulsingError::from(RuntimeError::Serialization(e.to_string())))?;
        Ok(Message::Single {
            msg_type: "SystemResponse".to_string(),
            data: json_data,
        })
    }
}

#[async_trait::async_trait]
impl Actor for SystemActor {
    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("type".to_string(), "SystemActor".to_string());
        meta.insert("builtin".to_string(), "true".to_string());
        meta.insert("path".to_string(), SYSTEM_ACTOR_PATH.to_string());
        meta.insert("control_plane".to_string(), "system-root".to_string());
        // Actor metadata is captured at spawn time, so advertise immutable
        // service identities here. Runtime readiness belongs to runtime@1.
        let service_ids = self
            .services
            .statuses()
            .into_iter()
            .map(|(manifest, _)| format!("{}@{}", manifest.id.namespace, manifest.id.major))
            .collect::<Vec<_>>()
            .join(",");
        meta.insert("services".to_string(), service_ids);
        meta
    }

    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> {
        let lifecycle = self.host.lifecycle();
        lifecycle.transition(NodeState::Starting)?;
        if let Err(error) = self.services.start_all().await {
            let _ = lifecycle.transition(NodeState::Failed);
            return Err(error);
        }
        lifecycle.transition(NodeState::Ready)?;
        tracing::info!(
            actor_id = ?ctx.id(),
            path = SYSTEM_ACTOR_PATH,
            "SystemActor started"
        );
        Ok(())
    }

    async fn on_stop(&mut self, ctx: &mut ActorContext) -> Result<()> {
        let lifecycle = self.host.lifecycle();
        lifecycle.begin_draining()?;
        if let Err(error) = self.services.stop_all().await {
            let _ = lifecycle.transition(NodeState::Failed);
            return Err(error);
        }
        tracing::info!(
            actor_id = ?ctx.id(),
            path = SYSTEM_ACTOR_PATH,
            "SystemActor stopped"
        );
        Ok(())
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> Result<Message> {
        self.metrics.inc_message();

        // Parse system message using auto-detection (JSON first, then bincode)
        let sys_msg: SystemMessage = match &msg {
            Message::Single { .. } => {
                match msg.parse() {
                    Ok(msg) => msg,
                    Err(e) => {
                        // Return error response instead of propagating error
                        return self.json_error_response(&format!("Invalid message format: {}", e));
                    }
                }
            }
            Message::Stream { .. } => {
                return self.json_error_response("Stream messages not supported by SystemActor");
            }
            Message::Tensor(_) => {
                return self.json_error_response("Tensor messages not supported by SystemActor");
            }
        };

        let response = match self
            .services
            .dispatch(
                service::SystemCommand::from_legacy(sys_msg),
                self.host.node_state(),
            )
            .await
        {
            Ok(response) => response,
            Err(error) => SystemResponse::Error {
                message: error.to_string(),
            },
        };

        // Use JSON serialization for response (for Python compatibility)
        let json_data = serde_json::to_vec(&response)
            .map_err(|e| PulsingError::from(RuntimeError::Serialization(e.to_string())))?;
        Ok(Message::Single {
            msg_type: "SystemResponse".to_string(),
            data: json_data,
        })
    }
}
