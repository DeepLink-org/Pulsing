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

mod factory;
mod messages;

pub use factory::{ActorFactory, BoxedActorFactory, DefaultActorFactory};
pub use messages::{ActorInfo, ActorStatusInfo, SystemMessage, SystemResponse};

use crate::actor::{Actor, ActorContext, ActorId, Message};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::metrics::SystemMetrics as PrometheusSystemMetrics;
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
        self.actors.insert(
            name.to_string(),
            ActorEntry {
                actor_id,
                actor_type: actor_type.to_string(),
                created_at: Instant::now(),
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
                metadata: std::collections::HashMap::new(), // TODO: get from actor
            })
            .collect()
    }

    pub fn get_info(&self, name: &str) -> Option<ActorInfo> {
        self.actors.get(name).map(|e| ActorInfo {
            name: name.to_string(),
            actor_id: e.actor_id.0,
            actor_type: e.actor_type.clone(),
            uptime_secs: e.created_at.elapsed().as_secs(),
            metadata: std::collections::HashMap::new(), // TODO: get from actor
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
    /// Local actor senders
    pub local_actors: Arc<DashMap<String, mpsc::Sender<crate::actor::Envelope>>>,
    /// Named actor path mappings
    pub named_actor_paths: Arc<DashMap<String, String>>,
}

/// SystemActor - Built-in system actor for each ActorSystem
pub struct SystemActor {
    /// Actor registry
    registry: Arc<ActorRegistry>,

    /// Actor factory
    factory: BoxedActorFactory,

    /// System metrics
    metrics: Arc<SystemMetrics>,

    /// System reference
    system_ref: Arc<SystemRef>,

    /// Start time
    start_time: Instant,
}

impl SystemActor {
    /// Create a new SystemActor
    pub fn new(system_ref: Arc<SystemRef>, factory: BoxedActorFactory) -> Self {
        Self {
            registry: Arc::new(ActorRegistry::new()),
            factory,
            metrics: Arc::new(SystemMetrics::new()),
            system_ref,
            start_time: Instant::now(),
        }
    }

    /// Create SystemActor with default factory
    pub fn with_default_factory(system_ref: Arc<SystemRef>) -> Self {
        Self::new(system_ref, Box::new(DefaultActorFactory))
    }

    /// Get registry (for Python bindings)
    pub fn registry(&self) -> &Arc<ActorRegistry> {
        &self.registry
    }

    /// Get metrics
    pub fn metrics(&self) -> &Arc<SystemMetrics> {
        &self.metrics
    }

    /// Handle ListActors request
    fn handle_list_actors(&self) -> SystemResponse {
        SystemResponse::ActorList {
            actors: self.registry.list_all(),
        }
    }

    /// Handle GetActor request
    fn handle_get_actor(&self, name: &str) -> SystemResponse {
        match self.registry.get_info(name) {
            Some(info) => SystemResponse::ActorInfo(info),
            None => SystemResponse::Error {
                message: format!("Actor not found: {}", name),
            },
        }
    }

    /// Handle GetMetrics request
    fn handle_get_metrics(&self) -> SystemResponse {
        SystemResponse::Metrics {
            actors_count: self.registry.count(),
            messages_total: self.metrics.messages_total(),
            actors_created: self.metrics.actors_created(),
            actors_stopped: self.metrics.actors_stopped(),
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }

    /// Handle GetNodeInfo request
    fn handle_get_node_info(&self) -> SystemResponse {
        SystemResponse::NodeInfo {
            node_id: self.system_ref.node_id.0,
            addr: self.system_ref.addr.to_string(),
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }

    /// Handle HealthCheck request
    fn handle_health_check(&self) -> SystemResponse {
        SystemResponse::Health {
            status: "healthy".to_string(),
            actors_count: self.registry.count(),
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }

    /// Handle Ping request
    fn handle_ping(&self) -> SystemResponse {
        SystemResponse::Pong {
            node_id: self.system_ref.node_id.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
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
        PrometheusSystemMetrics {
            node_id: self.system_ref.node_id.0,
            actors_count: self.registry.count(),
            messages_total: self.metrics.messages_total(),
            actors_created: self.metrics.actors_created(),
            actors_stopped: self.metrics.actors_stopped(),
            cluster_members: HashMap::new(), // Will be filled by caller with cluster info
        }
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
        meta
    }

    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> {
        tracing::info!(
            actor_id = ?ctx.id(),
            path = SYSTEM_ACTOR_PATH,
            "SystemActor started"
        );
        Ok(())
    }

    async fn on_stop(&mut self, ctx: &mut ActorContext) -> Result<()> {
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
        };

        let response = match sys_msg {
            SystemMessage::ListActors => self.handle_list_actors(),

            SystemMessage::GetActor { name } => self.handle_get_actor(&name),

            SystemMessage::GetMetrics => self.handle_get_metrics(),

            SystemMessage::GetNodeInfo => self.handle_get_node_info(),

            SystemMessage::HealthCheck => self.handle_health_check(),

            SystemMessage::Ping => self.handle_ping(),

            // Actor creation handled by Python layer (via extension)
            SystemMessage::CreateActor { .. } => SystemResponse::Error {
                message: "CreateActor not supported in pure Rust mode. Use Python extension."
                    .to_string(),
            },

            SystemMessage::StopActor { name } => {
                // Mark as stopped (actual stop handled by ActorSystem)
                if self.registry.contains(&name) {
                    self.unregister_actor(&name);
                    SystemResponse::Ok
                } else {
                    SystemResponse::Error {
                        message: format!("Actor not found: {}", name),
                    }
                }
            }

            SystemMessage::Extension { handler, payload } => {
                // Call factory's extension handler
                self.factory.handle_extension(&handler, payload).await
            }
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
